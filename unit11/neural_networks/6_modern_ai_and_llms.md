# Modern AI and Large Language Models

## Introduction
This guide connects traditional neural network concepts to modern AI applications, particularly Large Language Models (LLMs). We'll explore how fundamental concepts scale up to state-of-the-art AI systems.

## From Neural Networks to Transformers

### 1. Evolution of Architectures
```
Traditional NN → RNN → LSTM → Attention → Transformer → Modern LLMs
```

### 2. Key Innovations
- **Attention Mechanisms**: Allow models to focus on relevant parts of input
- **Self-Attention**: Enable parallel processing of sequences
- **Positional Encoding**: Maintain sequence order information
- **Multi-Head Attention**: Learn multiple relationship patterns

## Transformer Architecture

### 1. Basic Structure
```python
import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
```

### 2. Self-Attention Mechanism
```python
def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights
```

## Modern LLM Concepts

### 1. Pre-training and Fine-tuning
```python
# Example of fine-tuning setup
def create_fine_tuning_model(base_model, num_classes):
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add task-specific layers
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
    model = tf.keras.Model(base_model.input, outputs)
    return model
```

### 2. Prompt Engineering
```python
def format_prompt(instruction, context=None, examples=None):
    prompt = "Instruction: " + instruction + "\n\n"
    
    if context:
        prompt += "Context: " + context + "\n\n"
        
    if examples:
        prompt += "Examples:\n"
        for input_text, output_text in examples:
            prompt += f"Input: {input_text}\n"
            prompt += f"Output: {output_text}\n\n"
            
    prompt += "Input: "
    return prompt
```

## Scaling Considerations

### 1. Model Size and Computation
- Parameter counts in billions
- Training infrastructure requirements
- Inference optimization techniques

### 2. Memory Management
```python
def implement_gradient_checkpointing(model):
    """
    Implement gradient checkpointing to reduce memory usage
    during training of large models
    """
    if hasattr(model, 'layers'):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.MultiHeadAttention):
                layer._use_gradient_checkpointing = True
```

## Advanced Topics

### 1. Few-Shot Learning
```python
def few_shot_prompt(task, examples, new_input):
    prompt = f"Task: {task}\n\n"
    
    # Add examples
    for x, y in examples:
        prompt += f"Input: {x}\nOutput: {y}\n\n"
    
    # Add new input
    prompt += f"Input: {new_input}\nOutput:"
    return prompt
```

### 2. Efficient Fine-tuning Techniques
```python
class LoRALayer(tf.keras.layers.Layer):
    """
    Low-Rank Adaptation (LoRA) layer for efficient fine-tuning
    """
    def __init__(self, base_layer, rank=4, alpha=1):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.trainable = True
        
    def build(self, input_shape):
        self.lora_A = self.add_weight(
            "lora_A",
            shape=(input_shape[-1], self.rank),
            initializer="random_normal",
            trainable=True
        )
        self.lora_B = self.add_weight(
            "lora_B",
            shape=(self.rank, self.base_layer.units),
            initializer="zeros",
            trainable=True
        )
        
    def call(self, inputs):
        base_output = self.base_layer(inputs)
        lora_output = tf.matmul(
            tf.matmul(inputs, self.lora_A),
            self.lora_B
        ) * (self.alpha / self.rank)
        return base_output + lora_output
```

### 3. Ethical Considerations
- Bias in training data
- Model outputs responsibility
- Environmental impact
- Privacy concerns

## Best Practices for LLM Usage

### 1. Prompt Engineering
- Be specific and clear
- Provide context
- Use examples when needed
- Consider system context

### 2. Fine-tuning Guidelines
- Start with smaller models
- Use appropriate learning rates
- Monitor for catastrophic forgetting
- Implement evaluation metrics

### 3. Production Deployment
```python
def implement_model_serving(model, batch_size=32):
    """
    Example of model serving setup with batching
    """
    @tf.function(experimental_relax_shapes=True)
    def serve_fn(inputs):
        # Preprocess
        processed_inputs = preprocess_inputs(inputs)
        
        # Batch processing
        results = []
        for i in range(0, len(processed_inputs), batch_size):
            batch = processed_inputs[i:i + batch_size]
            results.append(model(batch))
            
        return tf.concat(results, axis=0)
    
    return serve_fn
```

## Common Challenges and Solutions

### 1. Handling Long Sequences
- Sliding window attention
- Sparse attention patterns
- Memory-efficient implementations

### 2. Reducing Computational Costs
- Quantization
- Knowledge distillation
- Model pruning

### 3. Improving Output Quality
- Temperature scaling
- Top-k/Top-p sampling
- Repetition penalties

## Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [The Handbook of Large Language Models](https://www.deeplearning.ai/the-batch/) 