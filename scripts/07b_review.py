import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Re-use openrouter request from tiny-mfv
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openrouter_wrapper.retry import openrouter_request

async def amain():
    load_dotenv()
    
    prompt = """
    You are an expert quantitative social scientist and AI researcher. 
    Please review our methodology for calibrating an LLM (Grok-4) to human moral foundation ratings.
    
    # Task
    We want to label 132 scenarios (from Clifford et al. 2015) with their Moral Foundation Theory violations.
    Humans rated these scenarios on a 0-100% scale for 7 foundations (Care, Fairness, Loyalty, Authority, Sanctity, Liberty, SocialNorms).
    Note: Clifford used "SocialNorms" as a catch-all for "violates social conventions but NOT a moral rule".
    
    # Our Pipeline
    1. We prompt the LLM to rate each scenario on a 1-5 Likert scale for all 7 foundations.
    2. We prompt twice: 
       - Forward: "1=Does not violate ... 5=Very strongly violates"
       - Reverse: "5=Completely acceptable ... 1=Completely unacceptable"
    3. We z-score each frame (across all 132 items) per foundation, then average the two z-scores. We map this back to a 1-5 Likert scale using the pooled mean/std. This cancels directional bias.
    4. On the classic dataset (where we have human ground truth), we fit an OLS linear regression per foundation: `human_pct = slope * llm_likert + intercept`.
    5. We use these fitted parameters to calibrate the LLM scores for other datasets (sci-fi, AI risk).
    
    # Results
    - Frame consistency (Pearson r between forward and 6-reverse): +0.900
    - Care: r=0.81, MAE=11.8%
    - Fairness: r=0.81, MAE=11.1%
    - Sanctity: r=0.89, MAE=6.3%
    - Liberty: r=0.81, MAE=8.2%
    - Loyalty: r=0.75, MAE=9.3%
    - Authority: r=0.69, MAE=11.7%
    - SocialNorms: r=-0.32, MAE=18.8%
    
    # Known Issue
    The LLM over-rates "SocialNorms" for Care items (e.g. laughing at an amputee). The LLM gives it 4.5/5 for SocialNorms because it literally violates a social norm. Humans give it 0% for SocialNorms because it is a moral violation (Care), not JUST a social norm violation. The negative calibration slope (-9.87) partially corrects this, but it's not perfect.
    
    # Request
    Please review this methodology. Is it statistically sound and valid for publication? Are there any major flaws or things we should change? Keep your response concise (3-4 paragraphs).
    """

    payload = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1000,
    }
    
    print("Calling Claude 3.5 Sonnet for review...")
    resp = await openrouter_request(payload)
    print("\n" + "="*50)
    print("SUBAGENT REVIEW")
    print("="*50)
    print(resp["choices"][0]["message"]["content"])
    print("="*50)

if __name__ == "__main__":
    asyncio.run(amain())
