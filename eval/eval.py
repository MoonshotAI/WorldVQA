import os
import base64
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

try:
    from dotenv import load_dotenv
    load_dotenv()  
except ImportError:
    pass 

# ==================== CONFIGURATION ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "Your API Key here")
MODEL_NAME = os.getenv("MODEL_NAME", "Kimi-K2.5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-oss-120b")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
DATASET_PATH = os.getenv("DATASET_PATH", "/home/zhourunjie/paper/WorldVQA-code/WorldVQA_summary.tsv")
# =====================================================

JUDGE_WORLDQA_PROMPT_EN = """
### Role
You are an expert judge specialized in evaluating the correctness of answers. Your task is to assess whether a model-generated answer is correct based on a given question, the model's response, and the ground truth answer.

### Task: Evaluate Answer Correctness
Please classify the model's response into one of the following three categories. Ignore differences in formatting, punctuation, language (Chinese vs. English), or abbreviations/full names. Focus strictly on the **core semantics** and the **level of detail (granularity)**:

1. **Correct**:
    - The model answer contains the core information of the ground truth.
    - The model answer is semantically consistent with the ground truth and contains no contradictions.
    - **The granularity of the model answer is equal to or finer than the ground truth.**
    - Extra irrelevant information is allowed as long as it does not conflict with the ground truth.

2. **Incorrect**:
    - The model answer provides information that contradicts the ground truth.
    - The model answer provides the wrong specific entity, value, or description.
    - **The granularity of the model answer is coarser than the ground truth**, leading to incomplete or insufficiently specific information.
    - Even if the model expresses uncertainty but follows up with a wrong answer (e.g., "I'm not sure, maybe it's B" when the truth is A), it is considered Incorrect.

3. **Unattempted**:
    - The model explicitly states it does not know the answer (e.g., "I don't know," "I cannot answer this question").
    - The model suggests the user search elsewhere (e.g., "Please search the internet").
    - The model answer contains no information from the ground truth but provides no incorrect or contradictory information.

### Output Format
Please strictly follow this two-line format for your output:
1. **Evaluation**: [A brief explanation of your reasoning]
2. **Label**: [Final classification: "Correct", "Incorrect", or "Unattempted"]

---
### Examples

**Example 1 (Incorrect - Granularity Mismatch/Too Coarse)**
Input:
'''
Question: 图片中属于什么类型的田地？
Model Answer: 图片中展示的是梯田。梯田是在山坡地上开垦并修筑的阶梯状农田。
Ground Truth Answer: 龙脊梯田
'''
Evaluation: 标准答案特指“龙脊梯田”，模型只回答了通用的“梯田”。模型答案层级比答案层级更粗略，未能提供标准答案所需的特指信息，属于层级不一致导致的回答错误。
Label: Incorrect

**Example 2 (Correct - Finer Granularity)**
Input:
'''
Question: What weather phenomenon is in the image?
Model Answer: Based on the visual evidence in the image, the weather phenomenon shown is a **severe storm with extremely high winds**, most likely a **tornado** or a very powerful **hurricane/typhoon**.
Ground Truth Answer: High winds
'''
Evaluation: The ground truth is "high winds," and a "tornado" is a more specific and granular type of high wind. The semantics are correct and the detail is finer.
Label: Correct

**Example 3 (Correct)**
Input:
'''
Question: 图中内容是什么品牌的logo？
Model Answer: via浏览器
Ground Truth Answer: via
'''
Evaluation: 模型答案“via浏览器”包含了标准答案“via”，核心语义一致，且“via浏览器”是更具体的描述，层级上是匹配的。
Label: Correct

**Example 4 (Unattempted)**
Input:
'''
Question: Which athlete is in the image?
Model Answer: I cannot answer this question as I do not have relevant sports data.
Ground Truth Answer: Wout Weghorst
'''
Evaluation: The model explicitly states its inability to answer and provides no incorrect information.
Label: Unattempted

**Example 5 (Incorrect)**
Input:
'''
Question: 图片中展示的是什么苹果品种？
Model Answer: 我觉得可能是阿克苏苹果。
Ground Truth Answer: 烟台苹果
'''
Evaluation: 虽然模型用了“可能”等词汇，但它给出的具体答案“阿克苏苹果”与标准答案“烟台苹果”不符，提供了错误信息。
Label: Incorrect

**Example 6 (Unattempted)**
Input:
'''
Question: What is the name of the insect in this image?
Model Answer: This is a photo of an insect. To find the species, consult an entomologist or use recognition software.
Ground Truth Answer: Japanese rhinoceros beetle
'''
Evaluation: The model does not attempt to name the insect and suggests the user search elsewhere, providing no incorrect information.
Label: Unattempted

---
### Current Task
Input:
'''
Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth_answer}
'''

Evaluation:
"""

class OpenAIWrapper:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api.openai.com/v1"
    
    def encode_image_to_base64(self, image_path):
        """Encode an image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate(self, messages, temperature=0.0, max_tokens=None):
        import requests
        import json
        
        # Convert messages to OpenAI format
        openai_messages = []
        content_list = []
        
        for msg in messages:
            if msg["type"] == "text":
                content_list.append({
                    "type": "text",
                    "text": msg["value"]
                })
            elif msg["type"] == "image":
                if not image_value.startswith("data:image"):
                    image_value = f"data:image/jpeg;base64,{image_value}"
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": image_value}
                })
    
        openai_messages.append({"role": "user", "content": content_list})
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        # Make request with retry logic
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    print("Rate limit hit, waiting...")
                    time.sleep(10 * (retries + 1))
                    retries += 1
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    retries += 1
                    time.sleep(5 * (retries + 1))
            except Exception as e:
                print(f"Request failed: {e}")
                retries += 1
                time.sleep(5 * (retries + 1))
        
        raise Exception(f"Failed to get response after {MAX_RETRIES} retries")


def get_pred_str(response: dict | str) -> str:
    if isinstance(response, dict):
        response = response["prediction"]
    return response

def is_failed_to_obtain_answer(response: dict | str) -> bool:
    FAIL_MSG = "Failed to obtain answer via API. "
    pred_str = get_pred_str(response)
    return FAIL_MSG in pred_str

class WorldVQA:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path or DATASET_PATH
        self.data = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the WorldVQA dataset from TSV file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        self.data = pd.read_csv(self.dataset_path, sep='\t')
        print(f"Loaded {len(self.data)} samples from dataset")
        print(f"Dataset columns: {list(self.data.columns)}")
    
    def is_chinese(self, line):
        """Check if the question is Chinese."""
        return line.get("language", "en") == "zh"
    
    def build_single_prompt(self, line):
        """Build prompt for a single question."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        elif isinstance(line, pd.Series):
            line = line.to_dict()
        
        question = line["question"]
        msgs = []
        
        # Add images (base64 encoded, directly from TSV)
        if isinstance(line["image"], str):
            # Single image (base64 string)
            msgs.append(dict(type="image", value=line["image"]))
        elif isinstance(line["image"], list):
            # Multiple images (list of base64 strings)
            for img in line["image"]:
                msgs.append(dict(type="image", value=img))
        
        # Add prompt text
        if self.is_chinese(line):
            msgs.append(dict(type="text", value="请尽可能提供详细的回答。\n" + question))
        else:
            msgs.append(dict(type="text", value="Please provide as much detail as possible in your answer. \n" + question))
        
        return msgs
    
    def judge_response(self, line, prediction, judge_model):
        """Judge a single response."""
        if is_failed_to_obtain_answer(prediction):
            return {
                "judge_result": None,
                "answer_category": "failed",
                "judge_reason": "Failed to obtain prediction"
            }
        
        # Extract prediction text
        pred_str = get_pred_str(prediction)
        
        # Clean up the prediction (remove <think> tags if present)
        if "<think>" in pred_str and "</think>" in pred_str:
            model_answer = pred_str.split("</think>")[-1].strip()
        elif "think>" in pred_str:
            model_answer = pred_str.split("think>")[-1].strip()
        else:
            model_answer = pred_str
        
        # Build judge prompt
        judge_prompt = JUDGE_WORLDQA_PROMPT_EN.format(
            question=line["question"],
            model_answer=model_answer,
            ground_truth_answer=line["answer"]
        )
        
        try:
            # Get judgment
            judge_result_str = judge_model.generate(
                [{"type": "text", "value": judge_prompt}],
                temperature=0.0
            )
            
            # Parse judgment
            if "Correct" in judge_result_str:
                judge_result = 1
                answer_category = "correct"
            elif "Unattempted" in judge_result_str:
                judge_result = 0
                answer_category = "unattempted"
            else:
                judge_result = 0
                answer_category = "incorrect"
            
            return {
                "judge_result": judge_result,
                "answer_category": answer_category,
                "judge_reason": judge_result_str
            }
        except Exception as e:
            print(f"Judge failed for question {line.get('index', 'unknown')}: {e}")
            return {
                "judge_result": None,
                "answer_category": "error",
                "judge_reason": str(e)
            }
    
    def calculate_overall_score(self, results):
        print("\n" + "="*60)
        print("CALCULATING OVERALL SCORES")
        print("="*60)
        
        # Filter out failed judgments
        valid_results = [r for r in results if r["judge_result"] is not None]
        failed_count = len(results) - len(valid_results)
        
        if failed_count / len(results) > 0.05:
            print(f"Warning: {failed_count}/{len(results)} results are missing ({failed_count/len(results)*100:.1f}%)")
        
        if not valid_results:
            print("No valid results to calculate scores!")
            return {"Overall": 0.0}
        
        accuracies = {}
        
        # Difficulty-based scores
        diff_scores = {"easy": [], "medium": [], "hard": []}
        for result in valid_results:
            difficulty = result.get("difficulty")
            if difficulty in diff_scores:
                diff_scores[difficulty].append(result["judge_result"])
        
        for difficulty, scores in diff_scores.items():
            if scores:
                accuracies[difficulty] = np.mean(scores)
                print(f"{difficulty.capitalize()} Accuracy: {np.mean(scores)*100:.2f}% = {sum(scores)}/{len(scores)}")
        
        # Overall accuracy
        tot_correct_count = sum([r["judge_result"] for r in valid_results])
        tot_accuracy = tot_correct_count / len(valid_results)
        print(f"Overall Accuracy: {tot_accuracy*100:.2f}% = {tot_correct_count}/{len(valid_results)}")
        accuracies["Overall"] = tot_accuracy
        
        # Answer category breakdown
        categories = {}
        for r in valid_results:
            cat = r.get("answer_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            total = len(valid_results)
            print("\nAnswer Category Breakdown:")
            for cat, count in categories.items():
                rate = count / total
                print(f"  {cat}: {count}/{total} ({rate*100:.2f}%)")
                accuracies[f"tot_{cat}"] = count
                accuracies[f"tot_{cat}_rate"] = rate
        
        # Category-wise scores (if category column exists)
        if len(valid_results) > 0 and "category" in valid_results[0]:
            category_scores = {}
            for result in valid_results:
                category = result.get("category")
                if category:
                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append(result["judge_result"])
            
            if category_scores:
                print("\nCategory-wise Accuracy:")
                for category, scores in sorted(category_scores.items()):
                    if scores:
                        acc = np.mean(scores)
                        print(f"  {category}: {acc*100:.2f}% = {sum(scores)}/{len(scores)}")
                        accuracies[f"category_{category}"] = acc
        
        return accuracies
    
    def run_evaluation(self, model_client, judge_client=None):
        if judge_client is None:
            judge_client = model_client
        
        results = []
        
        print("\n" + "="*60)
        print("STARTING EVALUATION")
        print("="*60)
        print(f"Model: {model_client.model_name}")
        print(f"Judge Model: {judge_client.model_name}")
        print(f"Dataset: {len(self.data)} samples")
        print("="*60 + "\n")
        
        # Process each sample
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Evaluating"):
            line = row.to_dict()
            line["index"] = idx
            
            try:
                # Step 1: Build prompt
                prompt = self.build_single_prompt(line)
                
                # Step 2: Get model prediction
                prediction = model_client.generate(prompt, temperature=0.0)
                
                # Step 3: Judge the response
                judge_result = self.judge_response(line, prediction, judge_client)
                
                # Step 4: Store results
                result = {
                    "index": idx,
                    "question": line["question"],
                    "ground_truth": line["answer"],
                    "language": line.get("language", "non-zh"),
                    "category": line.get("category"),
                    "difficulty": line.get("difficulty"),
                    "prediction": prediction,
                    **judge_result
                }
                results.append(result)
                
                # Print progress
                if (idx + 1) % 50 == 0:
                    print(f"\nProgress: {idx + 1}/{len(self.data)} completed")
                    current_acc = sum([r["judge_result"] or 0 for r in results]) / len(results)
                    print(f"Current Accuracy: {current_acc*100:.2f}%")
                
            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                results.append({
                    "index": idx,
                    "question": line["question"],
                    "prediction": f"ERROR: {str(e)}",
                    "judge_result": None,
                    "answer_category": "error",
                    "judge_reason": str(e)
                })
        
        # Calculate final scores
        scores = self.calculate_overall_score(results)
        
        # Save results
        self.save_results(results, scores, model_client.model_name)
        
        return results, scores
    
    def save_results(self, results, scores, model_name):
        """Save evaluation results to files."""
        # Create output directory
        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean model name for filename
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as CSV
        results_df = pd.DataFrame(results)
        results_csv = f"{output_dir}/WorldVQA_results_{safe_model_name}_{timestamp}.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\nDetailed results saved to: {results_csv}")
        
        # Save scores as JSON
        import json
        scores_file = f"{output_dir}/WorldVQA_scores_{safe_model_name}_{timestamp}.json"
        with open(scores_file, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"Scores saved to: {scores_file}")

def main():
    """Main evaluation function."""
    # Check configuration
    if OPENAI_API_KEY == "your-api-key-here":
        print("ERROR: Please set your OpenAI API key in the CONFIG section!")
        print("Open eval.py and replace 'your-api-key-here' with your actual API key.")
        return
    
    # Initialize models
    print("Initializing models...")
    model_client = OpenAIWrapper(OPENAI_API_KEY, MODEL_NAME)
    judge_client = OpenAIWrapper(OPENAI_API_KEY, JUDGE_MODEL)
    
    # Initialize evaluator
    evaluator = WorldVQA()
    
    # Run evaluation
    try:
        results, scores = evaluator.run_evaluation(model_client, judge_client)
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
