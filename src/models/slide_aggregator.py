import torch
import torch.nn as nn
from src.models.cnn import CustomCNN


class SlideAggregator(nn.Module):
    """
    A module to aggregate patch-level embeddings into a single slide-level embedding or prediction.

    Methods (set by `method`):
    - "mean": Mean pooling of patch embeddings (permutation-invariant)
    - "max": Max pooling of patch embeddings (permutation-invariant)
    - "topk": Top-K pooling of patch embeddings by L2 norm (permutation-invariant)
    - "deepset": A DeepSets architecture (permutation-invariant)
    - "gated_attention": Ilse et al. (2018) attention-based MIL (permutation-invariant)
    - "transformer_attention": Uses PyTorch's MultiheadAttention (permutation-sensitive)

    If `out_features` is specified, a final linear head produces classification logits of size [batch_size, out_features].

    Note: For "topk", "transformer_attention", you can adjust parameters like k=5, num_heads, etc. below.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg,
        # method: str = "gated_attention",
        embed_dim: int = None,
        out_features: int = None,
        use_transform: bool = True,
        dropout: float = 0.5,
        num_heads: int = 4,
        max_patches: int = 10,  # used if your number of patches is fixed
        k_top: int = 5,  # for topk pooling
        # logger = None,  # Optional logger for debugging
    ):
        """
        Args:
            model: A CNN or encoder that produces patch-level embeddings of size `embed_dim`.
            method: Aggregation method. One of ["mean", "max", "topk", "deepset", "gated_attention",
                                                "transformer_attention", "flat_mlp"].
            embed_dim: Size of the patch embeddings. If None, try to infer from `model`.
            out_features: If set, a final linear layer outputs this many features (e.g., # of classes).
            use_transform: Whether to apply a small transform (Linear->ReLU->Dropout) on patch embeddings.
            dropout: Dropout probability for transforms/attention layers.
            num_heads: Number of heads for multi-head attention (transformer_attention).
            max_patches: Used if you have a fixed number of patches and want to define positional embeddings
                         or flatten them in "flat_mlp".
            k_top: For top-k pooling, the number of top patches to average.

        """
        super().__init__()

        # Configurer le modèle
        if hasattr(model, "fc"):  # Pour ResNet
            self.embed_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif isinstance(model, CustomCNN):  # Pour CustomCNN
            self.embed_dim = model.feature_dim
            model.use_features = True  # Active le mode features
        else:
            if embed_dim is None:
                if hasattr(model, "vision_dim"):
                    self.embed_dim = model.vision_dim
                elif hasattr(model, "embed_dim"):
                    self.embed_dim = model.embed_dim
                elif hasattr(model, "num_features"):
                    self.embed_dim = model.num_features
                else:
                    raise ValueError("Cannot infer embed_dim from `model`. Please specify `embed_dim` manually.")
            else:
                self.embed_dim = embed_dim

        # Sauvegarder le modèle modifié
        self.model = model

        self.num_foVs = int(cfg.data.num_imgs_per_slide)

        # self.logger = logger
        self.cfg = cfg

        self.method = cfg.model.slide_aggregator_method.lower()
        if self.method == 'deepset':
            # self.deepset_aggregator = cfg.model.deepset.aggregation.lower()
            self.deepset_aggregator = "sum"
        self.use_transform = use_transform
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_patches = max_patches
        self.k_top = k_top
        self.out_features = out_features

        if self.use_transform:
            self.embedding_transform = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
        else:
            self.embedding_transform = nn.Identity()

        # --- Specific aggregator layers ---

        # 1) https://arxiv.org/pdf/1802.04712: Gated Attention (Ilse et al., 2018) - permutation invariant
        if self.method == "gated_attention":
            self.attention_V = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Sigmoid())
            self.attention_w = nn.Linear(self.embed_dim, 1)

            # Initialize
            nn.init.xavier_uniform_(self.attention_V[0].weight)
            nn.init.xavier_uniform_(self.attention_U[0].weight)
            nn.init.xavier_uniform_(self.attention_w.weight)
            nn.init.zeros_(self.attention_V[0].bias)
            nn.init.zeros_(self.attention_U[0].bias)
            nn.init.zeros_(self.attention_w.bias)

        # 2) https://arxiv.org/pdf/1703.06114: DeepSets aggregator - permutation invariant
        elif self.method == "deepset":
            self.phi = nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout),
            )
            self.rho = nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout),
                nn.Linear(512, self.embed_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout),
            )

        # 3) Transformer-style attention aggregator - permutation-invariant due to self-attention and no positional info
        elif self.method == "transformer_attention":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            self.self_attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(self.embed_dim)
        


        elif self.method not in ["mean", "max", "topk", "vote"]:
            raise ValueError(
                f"Unknown method {self.method}. Must be one of "
                f"['mean', 'max', 'topk', 'deepset', 'gated_attention', "
                f"'transformer_attention']."
            )

        # Classification head
        if out_features is not None:
            self.head = nn.Linear(self.embed_dim, out_features)
        else:
            self.head = None

    def forward(self, inputs):
        """
        Forward pass.
        images: [B, N, C, H, W]  or  [B, C, H, W] if only one patch
        """
        # Extract slide-level embeddings
        slide_embeddings = self.extract_features(inputs)
        # If a classification head is defined, apply it
        if self.head is not None:
            slide_embeddings = self.head(slide_embeddings)
        return slide_embeddings

    def extract_features(self, inputs):
        """
        Extract slide-level embeddings from patch-level images.
        """
        def _aggregate(embeddings):
            # Aggregate
            if self.method == "mean":
                return self.forward_mean(embeddings)
            elif self.method == "max":
                return self.forward_max(embeddings)
            elif self.method == "topk":
                return self.forward_topk(embeddings, self.k_top)
            elif self.method == "deepset":
                return self.forward_deepset(embeddings)
            elif self.method == "gated_attention":
                return self.forward_gated_attention(embeddings)
            elif self.method == "transformer_attention":
                return self.forward_transformer_attention(embeddings)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        if self.cfg.model.use_imgs_or_embeddings == 'embeddings':
            return _aggregate(inputs) 
        
        else: # inputs are images
            images = inputs  # images: [B, N, C, H, W] or [B, C, H, W]

            # If multiple patches per slide: [B, N, C, H, W]
            if images.dim() == 5:
                batch_size, num_patches = images.shape[:2]
                # Flatten patches for the base model
                images_flat = images.view(-1, *images.shape[2:])  # => [B*N, C, H, W]


                embeddings = self.model(images_flat)
                embeddings = embeddings.view(batch_size, num_patches, -1)  # => [B, N, embed_dim]

                return _aggregate(embeddings)  # => [B, embed_dim]

            else:
                # Single image only: [B, C, H, W]
                # Just extract features from model directly (no aggregation needed)
                return self.model(images)

    # -----------------
    # Aggregator methods
    # -----------------

    def forward_mean(self, embeddings):
        """
        Mean pooling: [B, N, D] -> [B, D]
        """
        embeddings = self.embedding_transform(embeddings)
        return embeddings.mean(dim=1)

    def forward_max(self, embeddings):
        """
        Max pooling: [B, N, D] -> [B, D]
        """
        embeddings = self.embedding_transform(embeddings)
        return embeddings.max(dim=1)[0]

    def forward_topk(self, embeddings, k=5):
        """
        Top-K pooling by L2 norm of each patch embedding:
        1) measure norm, 2) select top-k patches, 3) average them
        """
        embeddings = self.embedding_transform(embeddings)  # [B, N, D]
        norms = embeddings.norm(dim=-1)  # [B, N]
        _, topk_indices = norms.topk(k, dim=1)  # [B, k]
        batch_idx = torch.arange(embeddings.size(0), device=embeddings.device).unsqueeze(-1)
        # Gather top-k embeddings
        topk_embeddings = embeddings[batch_idx, topk_indices]  # [B, k, D]
        # Average the top-k
        return topk_embeddings.mean(dim=1)  # [B, D]

    def forward_deepset(self, embeddings):
        """
        DeepSets aggregator:
          - phi( x_i ) -> transform each patch embedding
          - sum over patches
          - rho( sum(...) ) -> final transform
        """

        # print(embeddings.shape)
        # print(self.embedding_transform)

        embeddings = self.embedding_transform(embeddings)  # [B, N, D]
        transformed = self.phi(embeddings)  # [B, N, 512]

        if self.deepset_aggregator == "sum":
            aggregated = transformed.sum(dim=1)
           

        elif self.deepset_aggregator == "mean":
            aggregated = transformed.mean(dim=1)


        elif self.deepset_aggregator == "max":
            aggregated, _ = transformed.max(dim=1)
        else:
            raise ValueError(f"Unknown deepset_pool: {self.deepset_aggregator}")

        # aggregated = transformed.sum(dim=1)  # [B, 512]

        return self.rho(aggregated)  # [B, D]

    def forward_gated_attention(self, embeddings):
        """
        Attention-based Deep MIL (Ilse et al., ICML 2018)
        - gating mechanism: a = w^T (tanh(Vh) ⨀ σ(Uh))
        """
        embeddings = self.embedding_transform(embeddings)  # [B, N, D]
        A_V = self.attention_V(embeddings)  # [B, N, D]
        A_U = self.attention_U(embeddings)  # [B, N, D]
        # elementwise product => [B, N, D]
        attention_logits = self.attention_w(A_V * A_U).squeeze(-1)  # [B, N]
        attention_weights = torch.softmax(attention_logits, dim=1)  # [B, N]
        # Weighted sum of embeddings
        return torch.sum(attention_weights.unsqueeze(-1) * embeddings, dim=1)  # [B, D]

    def forward_transformer_attention(self, embeddings):
        """
        Transformer-based aggregator:
         - prepend a learnable CLS token
         - self-attention among patches + CLS
         - output: [CLS] token's embedding
         (Permutation-sensitive, typically needs patch ordering or positional info)
        """
        batch_size = embeddings.size(0)

        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        embeddings = torch.cat([cls_token, embeddings], dim=1)  # [B, N+1, D]

        embeddings = self.embedding_transform(embeddings)  # [B, N+1, D]
        embeddings = self.norm(embeddings)

        attended = self.self_attn(embeddings, embeddings, embeddings)[0]  # [B, N+1, D]

        return attended[:, 0]  # [B, D]



