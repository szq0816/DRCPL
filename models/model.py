import math
import copy
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from utils.loss import triplet_loss_function, sdm_loss_function

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    try:
        # loading JIT archive
        model = torch.jit.load("/home/zgy/Code/DRCPL/clip/ViT-B-16.pt", map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load("/home/zgy/Code/DRCPL/clip/ViT-B-16.pt", map_location="cpu")

    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0,
                      "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model.float()


class ResMLP(torch.nn.Module):
    def __init__(self, emb_dimension, bottleneck_size, layer_norm=True, dropout=0.1, residual=True):
        super().__init__()
        layers = [nn.Linear(emb_dimension, bottleneck_size),
                  nn.ReLU(),
                  nn.Linear(bottleneck_size, emb_dimension)
                  ]

        if layer_norm:
            layers.append(nn.LayerNorm(emb_dimension))

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        self.module = torch.nn.Sequential(*layers)

        self.residual = residual

    def forward(self, inputs):
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        # third argument is the counter which denotes depth of prompt
        combined = [x, compound_prompts_deeper_text, 0]

        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_ctx = cfg.N_CTX
        ctx_init = cfg.CTX_INIT
        cfg_imsize = cfg.SIZE[0]

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        # Default is 1, which is compound shallow prompting
        assert cfg.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.PROMPT_DEPTH  # 9
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}".')
        print(f"Number of MaPLe context words (tokens): {n_ctx}.")
        print(f"Number of MaPLe prompt depth: {self.compound_prompts_depth}.")

        # These below, related to the shallow prompts
        self.ctx = nn.Parameter(ctx_vectors)

        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.sketch_first_proj = nn.Linear(ctx_dim, 768)
        self.image_first_proj = nn.Linear(ctx_dim, 768)

        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        # Also make corresponding projection layers, for each prompt
        sketch_layer = nn.Linear(ctx_dim, 768)
        self.sketch_compound_prompt_projections = self.get_clones(sketch_layer, self.compound_prompts_depth - 1)

        image_layer = nn.Linear(ctx_dim, 768)
        self.image_compound_prompt_projections = self.get_clones(image_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.res_mlp = cfg.RES_MLP
        if self.res_mlp:
            self.first_res_mlp = ResMLP(emb_dimension=ctx_dim, bottleneck_size=128,
                                        layer_norm=True, dropout=0.1, residual=True)

            compound_res_mlp = ResMLP(emb_dimension=ctx_dim, bottleneck_size=128,
                                      layer_norm=True, dropout=0.1, residual=True)
            self.compound_res_mlp = self.get_clones(compound_res_mlp, self.compound_prompts_depth - 1)
            print("W/ ResMLP.")
        else:
            print("W/O ResMLP.")

        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.n_cls = len(classnames)
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        if self.res_mlp:
            ctx = self.first_res_mlp(self.ctx)

            for index, params in enumerate(self.compound_prompts_text):
                self.compound_prompts_text[index].data = self.compound_res_mlp[index](params.data)
        else:
            ctx = self.ctx

        sketch_first_layer_prompts = self.sketch_first_proj(ctx)
        image_first_layer_prompts = self.image_first_proj(ctx)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [100, 2, 512]
        prefix = self.token_prefix
        suffix = self.token_suffix
        text_first_layer_prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        sketch_visual_deep_prompts = []
        for index, layer in enumerate(self.sketch_compound_prompt_projections):
            sketch_visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        image_visual_deep_prompts = []
        for index, layer in enumerate(self.image_compound_prompt_projections):
            image_visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        # Now the other way around
        # We will project the textual prompts from 512 to 768
        # pass here original, as for visual 768 is required
        return text_first_layer_prompts, sketch_first_layer_prompts, image_first_layer_prompts, \
            self.compound_prompts_text, sketch_visual_deep_prompts, image_visual_deep_prompts

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual

        self.cfg = cfg
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale
        self.sdm_logit_scale = torch.ones([]) * (1 / cfg.TEMPERATURE)

    def forward(self, sketch=None, image=None, label=None, split='train'):
        text_first_layer_prompts, sketch_first_layer_prompts, image_first_layer_prompts, \
            compound_prompts_text, sketch_visual_deep_prompts, image_visual_deep_prompts = self.prompt_learner()

        if self.prompt_learner.training and split == 'train':
            # calculate text features
            text_features = self.text_encoder(text_first_layer_prompts, self.tokenized_prompts, compound_prompts_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # calculate sketch features
            sketch_feature = self.image_encoder(sketch.type(self.dtype),
                                                sketch_first_layer_prompts,
                                                sketch_visual_deep_prompts
                                                )
            sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)

            # calculate image features
            image_feature = self.image_encoder(image.type(self.dtype),
                                               image_first_layer_prompts,
                                               image_visual_deep_prompts
                                               )
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            all_features = torch.cat([sketch_feature, image_feature], dim=0)

            # ------------------------------- calculate contrastive loss ----------------------------------
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * all_features @ text_features.t()

            loss_itc = F.cross_entropy(logits, label)

            # ---------------------------------- calculate triplet loss ----------------------------------
            image_pos, image_neg = torch.split(image_feature, self.cfg.BATCH, dim=0)

            loss_tri = triplet_loss_function(sketch_feature, image_pos, image_neg, args=self.cfg)

            # ----------------- calculate Similarity Distribution Matching loss --------------------------
            sk_label, im_pos_label, im_neg_label = torch.split(label, self.cfg.BATCH, dim=0)

            pos_loss_sdm = sdm_loss_function(sketch_features=sketch_feature,
                                             image_features=image_pos,
                                             sketch_pids=sk_label,
                                             image_pids=im_pos_label,
                                             logit_scale=self.sdm_logit_scale)

            neg_loss_sdm = sdm_loss_function(sketch_features=sketch_feature,
                                             image_features=image_neg,
                                             sketch_pids=sk_label,
                                             image_pids=im_neg_label,
                                             logit_scale=self.sdm_logit_scale)

            loss_sdm = pos_loss_sdm + neg_loss_sdm

            # ------------------------------- return losses ----------------------------------
            return loss_itc, loss_tri, loss_sdm
        else:
            if sketch is not None:
                sketch_feature = self.image_encoder(sketch.type(self.dtype),
                                                    sketch_first_layer_prompts,
                                                    sketch_visual_deep_prompts
                                                    )
                sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
                return sketch_feature
            elif image is not None:
                image_feature = self.image_encoder(image.type(self.dtype),
                                                   image_first_layer_prompts,
                                                   image_visual_deep_prompts
                                                   )
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                return image_feature
