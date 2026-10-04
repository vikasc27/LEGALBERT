import os
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pdfplumber
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

LABELS = ["FACTS", "ISSUE", "ARGUMENT", "ANALYSIS"]


class LegalSummarizer:
    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.similarity_model = None
        configured_path = model_path or os.getenv("LEGALBERT_MODEL_PATH", "model")
        self.model_path = configured_path
        self._load_models()

    def _load_models(self) -> None:
        try:
            required = ["config.json"]
            if not os.path.isdir(self.model_path) or any(
                not os.path.exists(os.path.join(self.model_path, name)) for name in required
            ):
                raise FileNotFoundError(f"LegalBERT model directory not found: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, local_files_only=True, num_labels=len(LABELS)
            ).to(self.device)
            self.model.eval()
            # The trained LegalBERT weights are loaded as-is; security stays outside
            # this service so classification and summarization remain unchanged.
        except Exception as error:
            print(f"LegalBERT unavailable; using rule-based fallback: {error}")
            self.model = None
        try:
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as error:
            print(f"Similarity model unavailable; using basic deduplication: {error}")
            self.similarity_model = None

    @staticmethod
    def create_overlapping_chunks(text: str, max_length: int = 400, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for index in range(0, len(words), max_length - overlap):
            chunk = " ".join(words[index:index + max_length]).strip()
            if len(chunk) > 20:
                chunks.append(chunk)
            if index + max_length >= len(words):
                break
        return chunks

    @staticmethod
    def advanced_sentence_split(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1\n\2", text)
        text = re.sub(
            r"(\bv\.|vs\.|Art\.|Sec\.|para\.|Para\.|cf\.|ibid\.|etc\.|e\.g\.|i\.e\.)",
            lambda match: match.group(1).replace(".", "<!DOT!>"), text,
        )
        sentences, current = [], ""
        for line in text.split("\n"):
            line = line.strip().replace("<!DOT!>", ".")
            if not line or len(line.split()) <= 2:
                continue
            if re.match(r"^\d+\.?\s*", line) or line.lower().startswith("page"):
                continue
            if current and not current.endswith((".", "!", "?")):
                current += " " + line
            else:
                if current:
                    sentences.append(current.strip())
                current = line
        if current:
            sentences.append(current.strip())
        return [
            sentence for sentence in sentences
            if 5 <= len(sentence.split()) <= 80
            and len(sentence) >= 25
            and sentence.count(".") <= 5
            and not re.match(r"^(page|chapter|section)\s+\d+", sentence.lower())
        ]

    def classify_with_model_chunked(self, sentence: str) -> Tuple[str, float]:
        try:
            chunks = self.create_overlapping_chunks(sentence) if len(sentence.split()) > 400 else [sentence]
            predictions = []
            confidences = []
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True,
                                        padding=True, max_length=512).to(self.device)
                with torch.no_grad():
                    probabilities = torch.softmax(self.model(**inputs).logits, dim=1)
                predicted = torch.argmax(probabilities, dim=1).item()
                predictions.append(predicted)
                confidences.append(probabilities[0][predicted].item())
            votes = defaultdict(float)
            for prediction, confidence in zip(predictions, confidences):
                votes[prediction] += confidence
            selected = max(votes, key=votes.get)
            return LABELS[selected], votes[selected] / len(predictions)
        except Exception:
            return self.classify_with_enhanced_rules(sentence)

    @staticmethod
    def classify_with_enhanced_rules(sentence: str) -> Tuple[str, float]:
        text = sentence.lower()
        patterns = {
            "FACTS": ["filed", "arrested", "charged", "occurred", "incident", "fir", "complaint", "investigation", "evidence", "witness", "appellant", "respondent", "accused", "victim", "petitioner", "prosecution", "police"],
            "ISSUE": ["whether", "question", "issue", "dispute", "interpretation", "jurisdiction", "constitutional", "legal question", "point of law", "violation of", "article", "section", "rule", "provision"],
            "ARGUMENT": ["submitted", "argued", "contended", "counsel", "advocate", "pleaded", "submission", "argument", "contention", "on behalf", "it is submitted", "learned counsel"],
            "ANALYSIS": ["held", "observed", "court found", "concluded", "judgment", "order", "decision", "ruling", "we find", "therefore", "hence", "accordingly", "this court", "supreme court", "high court"],
        }
        scores = {label: sum(3 if term in text else 0 for term in terms) for label, terms in patterns.items()}
        if re.search(r"\b(section|article|rule)\s+\d+", text):
            scores["ISSUE"] += 4
        if " vs. " in sentence.lower() or " v. " in sentence.lower():
            scores["FACTS"] += 4
        if re.search(r"\b(para|paragraph)\s+\d+", text):
            scores["ANALYSIS"] += 3
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        return best, min(scores[best] / total, 0.95) if scores[best] else 0.3

    def classify(self, sentence: str) -> Tuple[str, float]:
        if self.model is not None:
            return self.classify_with_model_chunked(sentence)
        return self.classify_with_enhanced_rules(sentence)

    def remove_similar_sentences(self, sentences: List[str], threshold: float = 0.8) -> List[str]:
        if not sentences or self.similarity_model is None:
            seen, unique = set(), []
            for sentence in sentences:
                normalized = sentence.strip().lower()
                if normalized not in seen and len(normalized) > 10:
                    unique.append(sentence)
                    seen.add(normalized)
            return unique
        try:
            embeddings = self.similarity_model.encode(sentences)
            matrix = np.dot(embeddings, embeddings.T)
            norms = np.linalg.norm(embeddings, axis=1)
            matrix /= np.outer(norms, norms)
            used, unique = set(), []
            for index in range(len(sentences)):
                if index in used:
                    continue
                unique.append(sentences[index])
                used.update(j for j in range(index + 1, len(sentences)) if matrix[index][j] > threshold)
            return unique
        except Exception:
            return list(dict.fromkeys(sentences))

    def smart_summarize(self, sentences: List[str], max_sentences: int = 5) -> List[str]:
        scored = []
        terms = ["court", "appellant", "respondent", "petitioner", "article", "section", "constitution", "law", "judgment", "order", "held", "observed", "contended", "argued"]
        for sentence in self.remove_similar_sentences(sentences, threshold=0.75):
            words = sentence.split()
            score = 3 if 10 <= len(words) <= 40 else 1 if 5 <= len(words) <= 60 else 0
            score += sum(term in sentence.lower() for term in terms)
            score += int(sentence.endswith((".", "?", "!")))
            scored.append((sentence, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        result = []
        for sentence, _ in scored:
            if len(result) >= max_sentences:
                break
            if all(self._word_overlap(sentence, chosen) < 0.6 for chosen in result):
                result.append(sentence)
        return result

    @staticmethod
    def _word_overlap(first: str, second: str) -> float:
        first_words, second_words = set(first.lower().split()), set(second.lower().split())
        return len(first_words & second_words) / len(first_words | second_words)

    @staticmethod
    def extract_fallback_content(full_text: str) -> Dict[str, List[str]]:
        fallback = {label: [] for label in LABELS}
        case_match = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)\s+v(?:s)?\.?\s+([A-Z][a-z]+ [A-Z][a-z]+)", full_text)
        if case_match:
            fallback["FACTS"].append(f"Case involves {case_match.group(1)} versus {case_match.group(2)}.")
        court_match = re.search(r"(Supreme Court|High Court|Sessions Court|District Court)", full_text, re.I)
        if court_match:
            fallback["FACTS"].append(f"Proceedings before the {court_match.group(1)}.")
        if re.search(r"article|section|constitution", full_text, re.I):
            fallback["ISSUE"].append("Constitutional or statutory questions involved.")
        if re.search(r"appellant|petitioner", full_text, re.I):
            fallback["ARGUMENT"].append("Appellant presented arguments against the lower court's decision.")
        if re.search(r"respondent|state", full_text, re.I):
            fallback["ARGUMENT"].append("Respondent defended the decision.")
        if re.search(r"judgment|held|disposed|concluded", full_text, re.I):
            fallback["ANALYSIS"].append("Court provided reasoning and final judgment.")
        return fallback

    def summarize_pdf(self, pdf_path: str) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if not full_text.strip():
            raise ValueError("No readable text found in PDF")
        classified = defaultdict(list)
        for sentence in self.advanced_sentence_split(full_text):
            classified[self.classify(sentence)[0]].append(sentence)
        fallback = self.extract_fallback_content(full_text)
        output = "LEGAL DOCUMENT SUMMARY\n" + "=" * 50 + "\n\n"
        for category in LABELS:
            points = self.smart_summarize(classified[category], max_sentences=5) if classified[category] else []
            points += [point for point in classified[category] if point not in points][:5 - len(points)]
            points = points[:5] or fallback[category] or [f"No explicit {category.lower()} content found."]
            output += f"{category}\n" + "-" * 30 + "\n" + "\n".join(f"{index}. {point}" for index, point in enumerate(points, 1)) + "\n\n"
        return output
