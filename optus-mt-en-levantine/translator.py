from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Translator:
    def __init__(self, model_path='../models/en_lev_ar_model/checkpoint-16650'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def translate(self, text, max_length=32, num_beams=4, length_penalty=0.6):
        inputs = self.tokenizer(text.lower().strip(),
                              return_tensors="pt",
                              max_length=max_length,
                              truncation=True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_batch(self, texts, batch_size=32, **kwargs):
        translations = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_translations = [self.translate(text, **kwargs) for text in batch]
            translations.extend(batch_translations)

        print(translations)

        return translations


translator = Translator()