import os
import json
import argparse
import logging
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class SAEInterpreter:
    def __init__(self, sae_data_dir, model_name, device='cuda', top_k=10, zero_k=20, random_k=10):
        self.sae_data_dir = sae_data_dir
        self.device = device
        self.top_k = top_k
        self.zero_k = zero_k
        self.random_k = random_k
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto'
        )
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.do_sample = False
        
        self.load_sae_data()
        
        self.interpreter_prompt = """
You are a meticulous AI and academic researcher conducting an important investigation into a certain neuron in a language model trained on Wikipedia articles. Your task is to figure out what sort of behavior this neuron is responsible for -- namely, on what general concepts, features, topics does this neuron fire? Here's how you'll complete the task:

INPUT_DESCRIPTION: 
You will be given two inputs: 1) Max Activating Examples and 2) Zero Activating Examples.

- MAX_ACTIVATING_EXAMPLES_DESCRIPTION
You will be given several examples of wikipedia abstract that activate the neuron, along with a number being how much it was activated (these number's absolute scale is meaningless, but the relative scale may be important). This means there is some feature, topic or concept in this text that 'excites' this neuron.

You will also be given several examples of abstract that don't activate the neuron. This means the feature, topic or concept is not present in these texts.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks. Be concise, and information dense. Don't waste a single word of reasoning.

Step 1: Based on the MAX_ACTIVATING_EXAMPLES provided, write down potential topics, concepts, and features that they share in common. These will need to be specific. You may need to look at different levels of granularity (i.e. subsets of a more general topic). List as many as you can think of. However, the only requirement is that all examples contain this feature.
Step 2: Based on the zero activating examples, rule out any of the topics/concepts/features listed above that are in the zero-activating examples. Systematically go through your list above.
Step 3: Based on the above two steps, perform a thorough analysis of which feature, concept or topic, at what level of granularity, is likely to activate this neuron. Use Occam's razor, the simplest explanation possible, as long as it fits the provided evidence. Opt for general concepts, features and topics. Be highly rational and analytical here.
Step 4: Based on step 3, summarise this concept in 1-8 words, in the form:
FINAL: [explanation]

Your final answer MUST end with "FINAL:" followed by your 1-8 word explanation. Do NOT write anything after the FINAL statement.

Here are the max-activating examples:

{max_activating_examples}

Here are the zero-activating examples:

{zero_activating_examples}

Work through the steps thoroughly and analytically to interpret our neuron.
"""
        
        self.prediction_prompt = """
You are an AI expert that is predicting which abstract will activate a certain neuron in a language model trained on Wikipedia articles. 
Your task is to predict which of the following abstract will activate the neuron the most. Here's how you'll complete the task:

INPUT_DESCRIPTION:
You will be given the description of the type of text on which the neuron activates. This description will be short.

You will then be given a abstract. Based on the concept of the text, you will predict whether the neuron will activate or not.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks.

Step 1: Based on the description of the type of wikipedia abstract on which the neuron activates, reason step by step about whether the neuron will activate on this abstract or not. Be highly rational and analytical here. The abstract may not be clear cut - it may contain topics/concepts close to the neuron description, but not exact. In this case, reason thoroughly and use your best judgement.
Step 2: Based on the above step, predict whether the neuron will activate on this abstract or not. If you predict it will activate, give a confidence score from 0 to 1 (i.e. 1 if you're certain it will activate because it contains topics/concepts that match the description exactly, 0 if you're highly uncertain). If you predict it will not activate, give a confidence score from -1 to 0.
Step 3: Provide the final confidence score in this exact format:
PREDICTION: [number]

Your response MUST end with "PREDICTION:" followed by your numerical score. Do NOT write anything after the numerical score.

Here is the description/interpretation of the type of wikipedia abstract on which the neuron activates:
{description}

Here is the abstract to predict:
{abstract}

Work through the steps thoroughly and analytically to predict whether the neuron will activate on this abstract.
"""

    def load_sae_data(self):
        print(f"Loading SAE data from {self.sae_data_dir}...")
        
        with open(os.path.join(self.sae_data_dir, "abstract_texts.json"), 'r') as f:
            self.abstract_texts = json.load(f)
        
        np_files = [f for f in os.listdir(self.sae_data_dir) if f.endswith('.npy')]
        feature_counts = set()
        
        for f in np_files:
            match = re.search(r'_(\d+)\.npy$', f)
            if match:
                feature_counts.add(int(match.group(1)))
        
        if not feature_counts:
            raise ValueError(f"Could not find feature count from files in {self.sae_data_dir}")
            
        self.feature_count = max(feature_counts)
        print(f"Detected feature count: {self.feature_count}")
        
        try:
            self.topk_indices = np.load(os.path.join(self.sae_data_dir, f"topk_indices_{self.top_k}_{self.feature_count}.npy"))
            self.topk_values = np.load(os.path.join(self.sae_data_dir, f"topk_values_{self.top_k}_{self.feature_count}.npy"))
            print(f"Loaded top-{self.top_k} activating examples")
        except FileNotFoundError as e:
            print(f"Warning: {e}. Will try to use default parameters.")
            self.topk_indices = np.load(os.path.join(self.sae_data_dir, f"topk_indices_5_2304.npy"))
            self.topk_values = np.load(os.path.join(self.sae_data_dir, f"topk_values_5_2304.npy"))
            self.top_k = 5
            
        try:
            self.zero_indices = np.load(os.path.join(self.sae_data_dir, f"zero_indices_{self.zero_k}_{self.feature_count}.npy"))
            self.zero_similarities = np.load(os.path.join(self.sae_data_dir, f"zero_similarities_{self.zero_k}_{self.feature_count}.npy"))
            print(f"Loaded {self.zero_k} zero-activating examples")
        except FileNotFoundError as e:
            print(f"Warning: {e}. Will try to use default parameters.")
            self.zero_indices = np.load(os.path.join(self.sae_data_dir, f"zero_indices_5_2304.npy"))
            self.zero_similarities = np.load(os.path.join(self.sae_data_dir, f"zero_similarities_5_2304.npy"))
            self.zero_k = 5
            
        try:
            self.random_indices = np.load(os.path.join(self.sae_data_dir, f"random_indices_{self.random_k}_{self.feature_count}.npy"))
            self.random_values = np.load(os.path.join(self.sae_data_dir, f"random_values_{self.random_k}_{self.feature_count}.npy"))
            print(f"Loaded {self.random_k} random-activating examples")
        except FileNotFoundError as e:
            print(f"Warning: Cannot find random indices and values. Will use top-k for testing: {e}")
            self.random_indices = self.topk_indices  
            self.random_values = self.topk_values
            self.random_k = self.top_k
            
        print(f"Loaded data for {self.topk_indices.shape[0]} features")

    def generate_text(self, prompt, max_new_tokens=2048):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens
            )
        
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def interpret_feature(self, feature_idx):
        num_top_examples = min(5, self.top_k // 2)
        max_activating_examples = ""
        
        for i in range(num_top_examples):
            if i >= self.topk_indices.shape[1]:
                break
            doc_idx = self.topk_indices[feature_idx, i]
            activation = self.topk_values[feature_idx, i]
            abstract = self.abstract_texts["abstracts"][doc_idx]
            max_activating_examples += f"Activation: {activation:.4f}\n{abstract}\n\n------------------------\n"
        
        num_zero_examples = min(5, self.zero_k // 2)
        zero_activating_examples = ""
        
        if self.zero_k > num_zero_examples:
            zero_sample_indices = np.random.choice(self.zero_k, num_zero_examples, replace=False)
        else:
            zero_sample_indices = np.arange(self.zero_k)
            
        for i in zero_sample_indices:
            if i >= self.zero_indices.shape[1]:
                break
            doc_idx = self.zero_indices[feature_idx, i]
            abstract = self.abstract_texts["abstracts"][doc_idx]
            zero_activating_examples += f"{abstract}\n\n------------------------\n"
        
        prompt = self.interpreter_prompt.format(
            max_activating_examples=max_activating_examples,
            zero_activating_examples=zero_activating_examples
        )
        
        response = self.generate_text(prompt)
        logging.info(f"Raw interpretation response: {response}")
        
        try:
            interpretation = response.split("FINAL:")[1].strip()
        except IndexError:
            logging.error(f"Failed to extract interpretation for feature {feature_idx}. Full response: {response}")
            if "FINAL" in response:
                for line in response.split('\n'):
                    if "FINAL" in line:
                        interpretation = line.split("FINAL")[1].strip(':').strip()
                        break
                else:
                    interpretation = "Unknown feature"
            else:
                interpretation = "Unknown feature"
        
        return interpretation

    def predict_activation(self, feature_interpretation, abstract):
        prompt = self.prediction_prompt.format(
            description=feature_interpretation,
            abstract=abstract
        )
        
        response = self.generate_text(prompt, max_new_tokens=1024)
        logging.info(f"Raw prediction response: {response}")

        try:
            prediction = float(response.split("PREDICTION:")[1].strip())
        except (IndexError, ValueError):
            logging.error(f"Failed to extract prediction. Full response: {response}")
            
            if "PREDICTION" in response:
                pattern = r"PREDICTION:\s*([-+]?\d*\.?\d+)"
                match = re.search(pattern, response)
                if match:
                    try:
                        prediction = float(match.group(1))
                    except ValueError:
                        prediction = 0.0
                else:
                    prediction = 0.0
            else:
                prediction = 0.0
        
        return prediction

    def evaluate_interpretation(self, feature_idx, interpretation):
        test_doc_ids = []
        test_abstracts = []
        ground_truth = []
        
        num_test_random = min(2, self.random_k)
        if self.random_k > num_test_random:
            rand_sample_indices = np.random.choice(self.random_k, num_test_random, replace=False)
        else:
            rand_sample_indices = np.arange(self.random_k)
        
        for i in rand_sample_indices:
            if i >= self.random_indices.shape[1]:
                continue
            doc_idx = self.random_indices[feature_idx, i]
            test_doc_ids.append(doc_idx)
            test_abstracts.append(self.abstract_texts["abstracts"][doc_idx])
            ground_truth.append(1)
        
        num_test_zero = min(2, self.zero_k) 
        if self.zero_k > num_test_zero:
            available_indices = set(range(self.zero_k))
            zero_test_indices = np.random.choice(list(available_indices), num_test_zero, replace=False)
        else:
            zero_test_indices = np.arange(self.zero_k)
            
        for i in zero_test_indices:
            if i >= self.zero_indices.shape[1]:
                continue
            doc_idx = self.zero_indices[feature_idx, i]
            test_doc_ids.append(doc_idx)
            test_abstracts.append(self.abstract_texts["abstracts"][doc_idx])
            ground_truth.append(0)
        
        if not test_abstracts:
            logging.error(f"No test abstracts found for feature {feature_idx}")
            return 0.0, 0.0, [], []
            
        predictions = []
        for abstract in test_abstracts:
            prediction = self.predict_activation(interpretation, abstract)
            predictions.append(prediction)
        
        if not predictions:
            logging.error(f"No predictions generated for feature {feature_idx}")
            return 0.0, 0.0, [], []
            
        binary_predictions = [1 if p > 0 else 0 for p in predictions]
        
        if len(set(ground_truth)) < 2 or len(set(binary_predictions)) < 2:
            f1 = 0.0 if set(ground_truth) != set(binary_predictions) else 1.0
        else:
            f1 = f1_score(ground_truth, binary_predictions)
            
        try:
            correlation, _ = pearsonr(ground_truth, predictions)
        except ValueError:
            correlation = float('nan')
        
        return f1, correlation, predictions, ground_truth

def main():
    parser = argparse.ArgumentParser(description="Interpret SAE features using LLMs")
    parser.add_argument("--sae_data_dir", type=str, default="./sae_data", help="Directory containing SAE data")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the LLM model to use")
    parser.add_argument("--feature_start", type=int, default=0, help="Start index for features to interpret")
    parser.add_argument("--feature_count", type=int, default=10, help="Number of features to interpret")
    parser.add_argument("--output_dir", type=str, default="./interpreted_features", help="Directory to save interpretations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top activating examples")
    parser.add_argument("--zero_k", type=int, default=20, help="Number of zero activating examples")  
    parser.add_argument("--random_k", type=int, default=10, help="Number of random activating examples")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "interpret.log")),
            logging.StreamHandler()
        ]
    )
    
    interpreter = SAEInterpreter(
        sae_data_dir=args.sae_data_dir,
        model_name=args.model_name,
        device=args.device,
        top_k=args.top_k,
        zero_k=args.zero_k,
        random_k=args.random_k
    )
    
    results = []
    for i in range(args.feature_start, args.feature_start + args.feature_count):
        logging.info(f"Interpreting feature {i}...")
        start_time = time.time()
        
        try:
            interpretation = interpreter.interpret_feature(i)
            logging.info(f"Feature {i} interpretation: {interpretation}")
            
            f1, correlation, predictions, ground_truth = interpreter.evaluate_interpretation(i, interpretation)
            logging.info(f"Feature {i} F1 score: {f1:.4f}, Pearson correlation: {correlation:.4f}")
            
            results.append({
                "feature_idx": i,
                "interpretation": interpretation,
                "f1_score": float(f1),
                "pearson_correlation": float(correlation),
                "predictions": [float(p) for p in predictions],
                "ground_truth": [int(g) for g in ground_truth]
            })
            
        except Exception as e:
            logging.error(f"Error interpreting feature {i}: {e}")
        
        logging.info(f"Feature {i} interpreted in {time.time() - start_time:.2f} seconds")
    
    with open(os.path.join(args.output_dir, "interpretations.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    if results:
        f1_scores = [r["f1_score"] for r in results]
        correlations = [r["pearson_correlation"] for r in results]
        logging.info(f"Interpreted {len(results)} features")
        logging.info(f"Average F1 score: {np.mean(f1_scores):.4f}")
        logging.info(f"Average Pearson correlation: {np.mean(correlations):.4f}")
    else:
        logging.warning("No features were successfully interpreted")

if __name__ == "__main__":
    main()