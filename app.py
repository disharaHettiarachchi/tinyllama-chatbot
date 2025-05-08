import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

@st.cache_resource
def load_model():
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "./tinyllama-finetuned-academic"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model

st.title("🎓 Academic Chatbot (TinyLlama)")

tokenizer, model = load_model()

prompt = st.text_area("Enter your academic prompt:", height=150)

if st.button("Generate Response"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=256)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.markdown("### 📚 Response:")
        st.write(response)
