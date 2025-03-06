from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

class Summarizer:
    def __init__(self, model_name="google/gemma-2-2b-it", max_position_embeddings=8192):
        logging.info("[Summarizer] Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_to_multiple_of = None  # Disable auto-padding
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.config.max_position_embeddings = max_position_embeddings
        logging.info("[Summarizer] Model and tokenizer loaded.")

    def summarize(self, conversation_text):
        logging.info("[Summarizer] Summarize called with text length: %d", len(conversation_text))
        try:
            # Tokenize with truncation only.
            inputs = self.tokenizer(
                conversation_text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=8192
            )
            # Generate summary with tuned parameters.
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                no_repeat_ngram_size=3,
                temperature=0.7,
                num_beams=4,
                do_sample=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logging.info("[Summarizer] Summarization result: %s", summary)
            print("[Summarizer][PRINT] Summarization result:", summary)
        except Exception as e:
            logging.error("[Summarizer] Exception during generation: %s", str(e))
            print("[Summarizer][PRINT] Exception during generation:", e)
            summary = ""
        # Flush logging handlers
        for handler in logging.getLogger().handlers:
            handler.flush()
        return summary
