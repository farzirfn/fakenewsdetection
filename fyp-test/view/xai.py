"""
Explainable AI (XAI) module for fake news detector
Provides attention visualization, token importance, and LIME explanations
"""

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from lime.lime_text import LimeTextExplainer
import streamlit as st


# ================================
# ATTENTION VISUALIZATION
# ================================
def extract_attention_weights(model, tokenizer, device, text):
    """Extract attention weights from the DistilBERT model"""
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)
        attention_weights = outputs.attentions
    
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    
    return attention_weights, tokens, encoding


def get_attention_heatmap(attention_weights, tokens, layer=0, head=0):
    """
    Extract and process attention weights for visualization
    
    Args:
        attention_weights: Tuple of attention weight tensors from model
        tokens: List of tokens from tokenizer
        layer: Which layer to visualize (default: last layer)
        head: Which attention head to visualize (default: first head)
    
    Returns:
        Plotly figure with attention heatmap
    """
    # Get the last layer if layer=-1
    if layer == -1:
        layer = len(attention_weights) - 1
    
    # Extract attention for the specified layer and head
    attn = attention_weights[layer][0, head].cpu().numpy()
    
    # Truncate to actual token length
    token_len = min(len(tokens), 50)  # Limit for visualization
    attn = attn[:token_len, :token_len]
    tokens_vis = tokens[:token_len]
    
    # Remove special tokens from display
    display_tokens = [t.replace('[CLS]', '[START]').replace('[SEP]', '[END]').replace('[PAD]', '') 
                     for t in tokens_vis]
    
    fig = go.Figure(data=go.Heatmap(
        z=attn,
        x=display_tokens,
        y=display_tokens,
        colorscale='Blues',
        colorbar=dict(title="Attention<br>Weight")
    ))
    
    fig.update_layout(
        title=f"Attention Weights (Layer {layer}, Head {head})",
        xaxis_title="Token",
        yaxis_title="Token",
        height=500,
        width=600,
        xaxis={'side': 'bottom'}
    )
    
    return fig


# ================================
# TOKEN IMPORTANCE (GRADIENT-BASED)
# ================================
def get_token_importance(model, tokenizer, label_encoder, device, text):
    """
    Calculate token importance using gradient-based method
    Shows which tokens contributed most to the prediction
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    
    # Clone tensor to track gradients
    input_ids = encoding['input_ids'].to(device).clone().detach().requires_grad_(False)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get embedding layer
    embeddings = model.distilbert.embeddings(input_ids)
    embeddings.requires_grad = True
    
    with torch.enable_grad():
        # Forward pass with embeddings
        output = model.distilbert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        logits = model.pre_classifier(output.last_hidden_state[:, 0])
        logits = model.classifier(logits)
        
        # Get the predicted class
        pred_class = torch.argmax(logits, dim=1)[0]
        loss = logits[0, pred_class]
        
        # Backward pass
        loss.backward()
    
    # Get gradient magnitudes
    grad_norms = torch.norm(embeddings.grad, dim=2)[0].detach().cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Get actual text tokens (remove padding and special tokens)
    token_list = []
    importance_list = []
    
    for idx, (token, importance) in enumerate(zip(tokens, grad_norms)):
        if token not in ['[PAD]', '[CLS]', '[SEP]'] and importance > 0:
            clean_token = token.replace('##', '')
            token_list.append(clean_token)
            importance_list.append(float(importance))
    
    return token_list, importance_list


def create_token_importance_chart(tokens, importances):
    """Create a bar chart of token importances"""
    # Sort by importance and get top tokens
    top_n = min(15, len(tokens))
    sorted_indices = np.argsort(importances)[-top_n:][::-1]
    
    top_tokens = [tokens[i] for i in sorted_indices]
    top_importances = [importances[i] for i in sorted_indices]
    
    fig = px.bar(
        x=top_importances,
        y=top_tokens,
        orientation='h',
        labels={'x': 'Importance Score', 'y': 'Token'},
        color=top_importances,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title="Top Tokens Contributing to Prediction",
        height=400,
        showlegend=False,
        xaxis_title="Importance Score",
        yaxis_title="Token"
    )
    
    return fig


# ================================
# LIME EXPLANATION
# ================================
def get_lime_explanation(model, tokenizer, label_encoder, device, text):
    """
    Generate LIME explanation for the prediction
    Shows which parts of text are important for the decision
    """
    
    def predict_proba(texts):
        """Prediction function for LIME"""
        results = []
        for t in texts:
            encoding = tokenizer(
                t,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt"
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = model(**encoding)
                probs = F.softmax(outputs.logits, dim=1)
            
            results.append(probs[0].cpu().numpy())
        
        return np.array(results)
    
    # Initialize LIME explainer
    explainer = LimeTextExplainer(
        class_names=label_encoder.classes_,
        verbose=False
    )
    
    # Get explanation
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=10,
        num_samples=50
    )
    
    return exp


def create_lime_visualization(exp, label_encoder):
    """Create a visualization of LIME explanation"""
    
    # Extract feature contributions
    features = []
    contributions = []
    
    for feature, weight in exp.as_list():
        features.append(feature)
        contributions.append(weight)
    
    # Separate positive and negative contributions
    colors = ['green' if c > 0 else 'red' for c in contributions]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=contributions,
            orientation='h',
            marker=dict(color=colors, opacity=0.7),
            text=[f"{c:.3f}" for c in contributions],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="LIME Explanation: Feature Contributions",
        xaxis_title="Contribution to Prediction",
        yaxis_title="Text Features",
        height=400,
        showlegend=False
    )
    
    return fig


# ================================
# COMBINED XAI REPORT
# ================================
def display_xai_report(model, tokenizer, label_encoder, device, text, prediction_label, confidence):
    """
    Display a comprehensive XAI report with multiple explanation methods
    """
    st.subheader("🔬 Explainable AI Analysis")
    st.info("Understanding why the model made this prediction:")
    
    # Create tabs for different explanation methods
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Token Importance",
        "🎯 Attention Visualization",
        "🔍 LIME Explanation",
        "📋 Summary"
    ])
    
    with tab1:
        st.markdown("### Which tokens influenced the decision?")
        try:
            tokens, importances = get_token_importance(model, tokenizer, label_encoder, device, text)
            if tokens and importances:
                fig = create_token_importance_chart(tokens, importances)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Higher importance scores indicate tokens that influenced the prediction more strongly")
            else:
                st.warning("Could not calculate token importance")
        except Exception as e:
            st.warning(f"Could not generate token importance: {str(e)}")
    
    with tab2:
        st.markdown("### How did the model attend to different tokens?")
        try:
            attention_weights, tokens, encoding = extract_attention_weights(model, tokenizer, device, text)
            
            # Allow selection of layer and head
            col1, col2 = st.columns(2)
            with col1:
                layer = st.slider("Select Layer", 0, len(attention_weights)-1, len(attention_weights)-1)
            with col2:
                head = st.slider("Select Head", 0, attention_weights[0].shape[1]-1, 0)
            
            fig = get_attention_heatmap(attention_weights, tokens, layer=layer, head=head)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows which tokens the model focused on during prediction (darker = higher attention)")
        except Exception as e:
            st.warning(f"Could not generate attention visualization: {str(e)}")
    
    with tab3:
        st.markdown("### Which text segments support this prediction?")
        try:
            exp = get_lime_explanation(model, tokenizer, label_encoder, device, text)
            
            fig = create_lime_visualization(exp, label_encoder)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show predicted probability
            pred_class_idx = list(label_encoder.classes_).index(prediction_label)
            pred_proba = exp.predict_proba[pred_class_idx]
            
            st.success(f"LIME Predicted {prediction_label}: {pred_proba:.2%}")
            st.caption("Green bars support the prediction, red bars contradict it")
        except Exception as e:
            st.warning(f"Could not generate LIME explanation: {str(e)}")
    
    with tab4:
        st.markdown("### 📌 Explanation Summary")
        
        summary = f"""
        **Prediction:** {prediction_label.upper()}  
        **Confidence:** {confidence*100:.2f}%
        
        **What you're seeing:**
        
        1. **Token Importance** - Shows which individual words/tokens the model considered most important for its decision
        
        2. **Attention Visualization** - Displays the internal attention mechanisms of the DistilBERT model. 
           Each layer and head has learned different attention patterns to focus on relevant parts.
        
        3. **LIME Explanation** - A model-agnostic approach showing which features (words/phrases) 
           support or contradict the prediction
        
        **How to interpret:**
        - Higher values = stronger influence on the prediction
        - Positive contributions support the prediction
        - Negative contributions argue against it
        
        **Note:** These visualizations reveal the model's decision-making process, helping identify 
        potential biases or interesting patterns in how the model evaluates news credibility.
        """
        
        st.markdown(summary)
