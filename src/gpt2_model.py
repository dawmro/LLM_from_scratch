


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module with causal masking for Transformer models.

    This implementation performs parallel self-attention across multiple heads using:
    1. A fused QKV projection for computational efficiency
    2. Multi-head decomposition for parallel attention computation
    3. Causal masking to prevent future context leakage in autoregressive models
    4. Final projection with residual connection and dropout

    The module follows the standard Transformer attention mechanism but with:
    - Fused QKV projection (single linear layer)
    - Causal masking for language modeling
    - Parallel attention heads with head sharing

    Args:
        n_embd (int): Total embedding dimension of input features
        num_heads (int): Number of attention heads to use
        block_size (int): Maximum sequence length for causal masking
        dropout (float): Dropout rate for attention weights and output projection
        qkv_bias (bool): Whether to include biases for Q, K, and V projections

    Attributes:
        num_heads (int): Number of attention heads
        head_size (int): Dimension of each attention head (n_embd // num_heads)
        qkv_proj (nn.Linear): Fused QKV projection layer (output dimension: 3 * n_embd)
        out_proj (nn.Linear): Final projection layer (output dimension: n_embd)
        attn_dropout (nn.Dropout): Dropout layer for attention weights
        resid_dropout (nn.Dropout): Dropout layer for final output projection
        tril (torch.Tensor): Causal mask buffer (block_size x block_size)
    """

    def __init__(self, n_embd: int, num_heads: int, block_size: int, dropout: float, qkv_bias: bool) -> None:
        """
        Initialize multi-head attention module with fused QKV projection.

        Args:
            n_embd (int): Dimension size of input embeddings
            num_heads (int): Number of parallel attention heads
            block_size (int): Maximum sequence length for causal masking
            dropout (float): Dropout probability for regularization
            qkv_bias (bool): Whether to include biases for Q, K, and V projections  
            
        Raises:
            AssertionError: If n_embd is not divisible by num_heads
            
        Notes:
            - Uses fused QKV projection (3*n_embd) instead of separate projections
            - head_size is automatically computed as n_embd // num_heads
            - Causal mask is registered as buffer for efficient sequence length slicing
        """
        super().__init__()

        # Validate head count compatibility
        assert n_embd % num_heads == 0, "Input embedding dimension must be divisible by number of heads"

         # Store configuration
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads  # Automatically compute head size
        self.dropout = dropout

        # Fused QKV Projection
        # Projects input embeddings to concatenated [Q; K; V] vectors of size 3*n_embd
        # This is more efficient than separate projections due to better memory locality
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=qkv_bias)

        # Final Output Projection
        # Projects concatenated multi-head outputs back to original embedding dimension
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False) # No bias in output projection

        # Regularization Layers
        # Attention dropout: applied to softmax probabilities
        self.attn_dropout = nn.Dropout(dropout)
        # Residual dropout: applied after final projection
        self.resid_dropout = nn.Dropout(dropout)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention output for input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, n_embd)
                - batch_size (B): Number of sequences in batch
                - seq_length (T): Length of each sequence
                - n_embd (C): Embedding dimension

        Returns:
            Tensor: Output tensor of same shape (B, T, C) after attention and projection
        """
        # Get input dimensions
        B, T, C = x.size()

        # 1. Fused QKV Projection
        # Input: (B, T, C) -> Output: (B, T, 3*C)
        # Projects to concatenated [Q; K; V] vectors
        qkv = self.qkv_proj(x)

        # 2. Split into Query, Key, Value
        # Each projection is (B, T, C) where C = n_embd
        q, k, v = qkv.chunk(3, dim=2)

        # 3. Reshape for Multi-Head Attention
        # (B, T, C) -> (B, heads, T, head_size)
        # 1. View: (B, T, C) -> (B, T, heads, head_size)
        # 2. Transpose: (B, T, heads, head_size) -> (B, heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # using FlashAttention for steps 4. to 8.
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )

        # 9. Concatenate Attention Heads
        # (B, heads, T, head_size) -> (B, T, heads*head_size) = (B, T, C)
        # Use contiguous() before view() to ensure memory layout compatibility
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 10. Final Output Projection
        # (B, T, C) -> (B, T, C)
        # Apply residual dropout before returning    
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """A feed-forward neural network layer with bottleneck architecture for Transformer.

    This module performs a linear transformation followed by a non-linear activation, 
    then another linear projection back to the original dimension with optional dropout.

    Args:
        n_embd (int): Input and output embedding dimension (the blocks are designed to be residual).
        dropout (float): Dropout rate for regularization. 

    Attributes:
        net (nn.Sequential): A neural network sequence comprising:
            1. A linear layer expanding to 4x input dimension.
            2. ReLU activation for non-linearity.
            3. A linear layer projecting back to original dimension.
            4. Dropout for regularization.
    """
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        # Feeds through a ReLU-activated bottleneck layer with 4x expansion
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),      # Expand input to 4x width
            nn.GELU(),                          # Non-linear activation
            nn.Linear(4 * n_embd, n_embd),      # Back to input dimension
            nn.Dropout(dropout)                 # Regularization 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying the feed-forward sequence.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, n_embd)
            
        Returns:
            Tensor: Output tensor with residual connection applied (same shape as input).
        """
        return self.net(x) # Directly applies the sequential network layers

    
class Block(nn.Module):
    """
    Transformer block combining self-attention and feed-forward layers with residual connections.

    Implements the core Transformer architecture with post-norm (normalization applied after residual connections). 

    Args:
        n_embd (int): Dimension of input embeddings.
        n_head (int): Number of self-attention heads.
        block_size (int): Maximum context length (required for causal masking).
        dropout (float): Dropout rate for attention and feed-forward layers.
        qkv_bias (bool): Whether to include biases for Q, K, and V projections

    Attributes:
        sa (MultiHeadAttention): Multi-head self-attention layer with causal masking.
        ffwd (FeedForward): Feed-forward neural network subnet.
        ln1 (nn.LayerNorm): First layer normalization to stabilize training.
        ln2 (nn.LayerNorm): Second layer normalization after feed-forward layer.
    """
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, qkv_bias: bool, use_checkpoint: bool = False) -> None:
        super().__init__()

        self.use_checkpoint = use_checkpoint

        # Validate head count compatibility
        assert n_embd % n_head == 0, "Input embedding dimension must be divisible by number of heads"

        self.sa = MultiHeadAttention(    # Initialize causal self-attention layer
            n_embd=n_embd,
            num_heads=n_head,
            block_size=block_size,
            dropout=dropout,
            qkv_bias=qkv_bias
        )
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)  # Feed-forward subnet
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)         # Normalization after attention, Match GPT-2 spec
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)         # Normalization after feed-forward, Match GPT-2 spec
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute Transformer block computation with post-norm architecture.

        Args:
            x (Tensor): Input tensor of shape [batch_size, sequence_length, n_embd]

        Steps:
            1. Self-Attention: Compute attention with residual and LayerNorm.
            2. Feed-Forward: Apply non-linear processing with residual and LayerNorm.
            
        Returns:
            Tensor: Transformer-block processed tensor (same shape as input).
        """

        # vvvvvvvvvvvv torch.utils.checkpoint  vvvvvvvvvvvvvvvvvvvv
        if self.training and self.use_checkpoint:
            # 1. Self-Attention sublayer with checkpointing to save memory
            # Compute multi-head attention
            #sa_out = checkpoint(self.sa, x) # Post-Norm
            sa_out = checkpoint(self.sa, self.ln1(x))  # Pre-Norm before checkpointed attention
        else:
            # directly call without checkpoint in eval mode
            #sa_out = self.sa(x) # Post-Norm
            sa_out = self.sa(self.ln1(x)) # Pre-Norm
        # Add residual connection + normalize
        x = x + sa_out # Pre-Norm
        # x = self.ln1(x + sa_out) # Post-Norm

        if self.training and self.use_checkpoint:
            # 2. Feed-Forward sublayer with checkpointing
            # Non-linear bottleneck
            # ffwd_out = checkpoint(self.ffwd, x) # Post-Norm 
            ff_out = checkpoint(self.ffwd, self.ln2(x))  # Pre-Norm before checkpointed FF
        else:
            # directly call without checkpoint in eval mode
            # ffwd_out = self.ffwd(x) # Post-Norm  
            ff_out = self.ffwd(self.ln2(x)) # Pre-Norm
        # Residual + normalization
        x = x + ff_out # Pre-Norm
        # x = self.ln2(x + ffwd_out) # Post-Norm  
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        return x


class GPTLanguageModel(nn.Module):
    """A full autoregressive language model with multi-head self-attention and positional embeddings.

    This model combines token and positional embeddings with a sequence of self-attention blocks to predict the next token in a sequence.

    Features:
    - Token and positional embeddings
    - Multi-head self-attention with causal masking
    - Layer normalization and residual connections
    - Weight tying between input and output embeddings
    - Advanced generation with sampling techniques

    Args:
        vocab_size: Number of tokens in vocabulary
        n_embd: Embedding dimension size
        n_head: Number of attention heads
        block_size: Maximum context length
        n_layer: Number of transformer blocks
        dropout: Dropout probability
        device: Computation device ('cpu' or 'cuda')
        qkv_bias: Enable bias for QKV projections
        ignore_index: Target value to ignore in loss calculation
    
    
    Attributes:
        vocab_size (int): Size of the token vocabulary.
        n_embd (int): Dimensionality of token and positional embeddings.
        n_head (int): Number of self-attention heads.
        block_size (int): Maximum context length the model can handle.
        n_layer (int): Number of transformer blocks in the network.
        device (str): Device for tensor operations ('cpu' or 'cuda').
        token_embedding_table (nn.Embedding): Maps tokens to embedding vectors.
        position_embedding_table (nn.Embedding): Encodes positional information up to block_size positions.
        blocks (nn.Sequential): Sequence of transformer blocks processing embeddings.
        ln_f (nn.LayerNorm): Final layer normalization after transformer blocks.
        lm_head (nn.Linear): Linear layer predicting next-token probabilities.
    """
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        block_size: int,
        n_layer: int,
        dropout: float,
        device: str,
        qkv_bias: bool = False,       
        ignore_index: int = -100
    ) -> None:
        """Initialize the GPT-style language model.
        
        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            n_embd (int): Dimension of token and positional embeddings (output size of embeddings).
            n_head (int): Number of parallel attention heads.
            block_size (int): Maximum sequence length the model can process.
            n_layer (int): Number of transformer blocks in the neural network.
            dropout (float): Dropout probability applied in attention and feedforward layers.
            device (str): Execution device ('cuda' for GPU acceleration).
            qkv_bias (bool): Whether to include biases for Q, K, and V projections (default: False)
            ignore_index (int): Label value to ignore in loss calculation (default: -100).
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device
        self.n_layer = n_layer

        # Create position buffer for safe indexing
        self.register_buffer('position_ids', torch.arange(block_size))

        # 1) Embedding layers map input indices to dense vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2) how many indices of size n_enbd
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # 3) Sequential transformer blocks
        # how many decoder blocks running sequentially
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout, qkv_bias) 
            for _ in range(n_layer)
        ])

        # 4) Final processing layers
        self.ln_f = nn.LayerNorm(n_embd) # Final layer normalization stabilizes training
        # 5) Output head with/without bias
        # This projects from hidden_dim → vocab_size
        # Predict next token probabilities and always include bias
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True) 

        # Optional
        # 6) Tie weights: Share weights between input and output embeddings
        #    Now lm_head.weight and token_embedding_table.weight are the same tensor
        #    Reduces parameters and improves convergence
        self.lm_head.weight = self.token_embedding_table.weight

        # 7) Initialize weights using standard transformer initialization strategy
        self.apply(self._init_weights)
        self.to(device) # Move all components to the specified device

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using heuristics from the original Transformer paper.
        
        Args:
            module (nn.Module): Module to be initialized (linear/embedding layers).
        """
        if isinstance(module, nn.Linear):
            # Weight initialization tailored for deep networks (Gaussian)
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Bias terms initialized to zero as per best practices
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding weights initialized with small random values
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute logit predictions and optionally calculate loss.
        
        Args:
            input_tokens (Tensor): Input tensor of shape (batch_size, context_length)
            targets (Tensor, optional): Target tokens of shape (batch_size, context_length)
        
        Returns:
            logits (Tensor): Unnormalized token probabilities (batch_size, context_length, vocab_size)
            loss (Tensor, optional): Loss value for training if targets provided
        """
        # Batch size and current sequence length
        B,T = input_tokens.shape

        # Validate sequence length
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        # 1. Token embeddings (B, T) → (B, T, C)
        # index and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(input_tokens) # (B,T,C)

        # 2. Positional embeddings (T) → (T, C)
        # Positional embeddings using safe buffer indexing
        # Avoids creating new arange tensor each forward pass
        pos = self.position_ids[:T].to(input_tokens.device)
        pos_emb = self.position_embedding_table(pos)

        # 3. Combine embeddings with element-wise addition → (B, T, C) and apply dropout
        x = self.drop(tok_emb + pos_emb) # (B,T,C) with dropout

        # 4. Pass through transformer blocks → (B, T, C)
        x = self.blocks(x) # (B,T,C)

        # 5. Final layer normalization → (B, T, C)
        x = self.ln_f(x) # (B,T,C)

        # 6. Compute next-token predictions → (B, T, vocab_size)
        logits = self.lm_head(x) # (B,T,vocab_size)

        # Loss calculation (only done during training)
        loss = None
        if targets is not None:
            # Reshape to merge batch and time dimensions for cross-entropy (B*T, vocab_size)
            B,T,C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat  = targets.view(B*T)

            # Calculate cross-entropy loss ignoring padding tokens
            loss = F.cross_entropy(logits_flat, targets_flat , ignore_index=self.ignore_index)
            
        return logits, loss
   
    def generate(
        self, 
        input_tokens: torch.Tensor, 
        max_new_tokens: int, 
        **kwargs
    ) -> torch.Tensor:
        """Unified generation interface with sampling support.
        
        Args:
            input_tokens: (B, T) starting sequence
            max_new_tokens: Number of tokens to generate
            **kwargs: Sampling parameters passed to advanced_generation
            
        Returns:
            Generated sequence (B, T + max_new_tokens)
        """
        return self.advanced_generation(
            input_tokens, 
            max_new_tokens, 
            temperature=1.0, 
            **kwargs
        )
    
    def advanced_generation(self,
        input_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: int = 50256
    ) -> torch.Tensor:
        """Generate tokens with advanced sampling strategies (temperature, nucleus, top-k).
        
        Args:
            input_tokens (Tensor): Initial tokens of shape (B, T)
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Controls randomness (1=default, 0→deterministic)
            top_k (int, optional): Keep only top K most probable tokens (None→disabled)
            top_p (float, optional): Keep smallest window ≥ cumulative prob mass p (None→disabled)
        
        Returns:
            Tensor: Generated sequence (B, T + max_new_tokens)
        """

        # Validate sampling parameters
        if temperature < 0:
            raise ValueError("Temperature must be >= 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive integer or None")
        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError("top_p must be in (0, 1] range")
        
        for _ in range(max_new_tokens):
            # Crop to block_size with efficient slicing
            cropped_input = input_tokens[:, -self.block_size:]

            # Get predictions without loss calculation
            logits, _ = self(cropped_input)

            # Only consider last predicted token
            next_logits = logits[:, -1, :] 

            # Handle greedy sampling separately for efficiency
            if temperature == 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                input_tokens = torch.cat((input_tokens, next_token), dim=1)
                continue

            # Temperature scaling controls exploration-exploitation tradeoff
            scaled_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                # Find top-k values (dynamic k for small vocabularies)
                k = min(top_k, scaled_logits.size(-1))
                top_k_vals = torch.topk(scaled_logits, k, dim=-1).values
                # Create mask and apply
                mask = scaled_logits < top_k_vals[:, [-1]]
                scaled_logits = scaled_logits.masked_fill(mask, -float('inf'))

            # Convert to probabilities after any pre-processing
            probs = F.softmax(scaled_logits, dim=-1)

            # Apply nucleus (top-p) sampling (if requested)
            if top_p is not None:
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Determine indices to mask by identifying positions exceeding p
                sorted_indices_to_remove = cumulative_probs > top_p

                # Keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Create original-order mask
                remove_mask = torch.zeros_like(probs, dtype=torch.bool)
                remove_mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)

                # Filter and renormalize
                probs = probs.masked_fill(remove_mask, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample with controlled probability distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Extend sequence
            input_tokens = torch.cat((input_tokens, next_token), dim=1)

            # If we generated EOS, stop immediately
            if next_token.item() == eos_token_id:
                break

        return input_tokens  # Final expanded tensor