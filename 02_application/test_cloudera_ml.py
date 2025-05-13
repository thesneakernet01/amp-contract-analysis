# test_cloudera_ml.py
import json
import os
import sys

# Uncomment to see detailed debug information
# import logging
# logging.basicConfig(level=logging.DEBUG)

print("Testing Cloudera ML connection with litellm...")

# Try to load the JWT token
try:
    print("Attempting to load CDP token from /tmp/jwt...")
    with open("/tmp/jwt", "r") as f:
        token_data = json.load(f)
        if "access_token" in token_data:
            API_KEY = token_data["access_token"]
            print("✅ CDP token loaded successfully")
        else:
            print("❌ Failed to find access_token in /tmp/jwt")
            API_KEY = input("Please enter your API key/CDP token: ")
except Exception as e:
    print(f"❌ Error loading CDP token: {e}")
    API_KEY = input("Please enter your API key/CDP token: ")

# Get endpoint from user input or use default
API_ENDPOINT = input("Enter your Cloudera ML endpoint URL (press Enter to use default): ")
if not API_ENDPOINT:
    API_ENDPOINT = "https://ml-2dad9e26-62f.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/contract-analysis-inference/v1"
    print(f"Using default endpoint: {API_ENDPOINT}")

# Get model from user input or use default
MODEL_ID = input("Enter your model ID (press Enter to use meta/llama-3.1-8b-instruct): ")
if not MODEL_ID:
    MODEL_ID = "meta/llama-3.1-8b-instruct"
    print(f"Using default model ID: {MODEL_ID}")

# Set environment variables
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = API_ENDPOINT

print("\n--- Test 1: Direct OpenAI API call ---")
try:
    from openai import OpenAI

    print(f"Creating OpenAI client with base_url={API_ENDPOINT}")
    client = OpenAI(
        base_url=API_ENDPOINT,
        api_key=API_KEY,
    )

    print(f"Making API call with model={MODEL_ID}")
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Say 'Hello from Cloudera ML!'"}],
        temperature=0.2,
        max_tokens=30
    )

    response_text = completion.choices[0].message.content
    print(f"✅ Success! Response: {response_text}")
except Exception as e:
    print(f"❌ Error with direct OpenAI API call: {e}")

print("\n--- Test 2: litellm with no provider prefix ---")
try:
    import litellm

    print(f"Making litellm call with model={MODEL_ID}")
    response = litellm.completion(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Say 'Hello from litellm with no provider!'"}],
        temperature=0.2,
        max_tokens=30
    )

    response_text = response.choices[0].message.content
    print(f"✅ Success! Response: {response_text}")
except Exception as e:
    print(f"❌ Error with litellm (no provider): {e}")

print("\n--- Test 3: litellm with custom provider prefix ---")
try:
    # Try with different provider prefixes
    for provider in ["cloudera", "llama", "custom", "huggingface"]:
        provider_model = f"{provider}/{MODEL_ID}"
        print(f"Making litellm call with model={provider_model}")

        try:
            response = litellm.completion(
                model=provider_model,
                messages=[{"role": "user", "content": f"Say 'Hello from litellm with {provider} provider!'"}],
                temperature=0.2,
                max_tokens=30
            )

            response_text = response.choices[0].message.content
            print(f"✅ Success with {provider} provider! Response: {response_text}")
            # If this succeeds, we found the right provider
            break
        except Exception as e:
            print(f"❌ Error with {provider} provider: {e}")
except Exception as e:
    print(f"❌ Error setting up litellm tests: {e}")

print("\n--- Test 4: langchain_openai with ChatOpenAI ---")
try:
    from langchain_openai import ChatOpenAI

    print(f"Creating ChatOpenAI with model={MODEL_ID}")
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=API_ENDPOINT,
        model=MODEL_ID,
        temperature=0
    )

    print("Getting response from ChatOpenAI")
    response = llm.invoke("Say 'Hello from langchain_openai!'")
    print(f"✅ Success! Response: {response.content}")
except Exception as e:
    print(f"❌ Error with langchain_openai: {e}")

print("\n--- Test 5: Patched litellm ---")
try:
    import litellm

    # Store the original completion function
    original_completion = litellm.completion


    def patched_completion(*args, **kwargs):
        # If model is provided but doesn't have a provider prefix
        if 'model' in kwargs:
            model = kwargs['model']
            # Add cloudera provider prefix
            if not any(p in model for p in ["cloudera/", "llama/", "custom/", "huggingface/"]):
                # Add provider prefix to model
                kwargs['model'] = f"cloudera/{model}"
                print(f"Patched litellm call: Added provider prefix to model: {kwargs['model']}")

        # Call the original function with modified arguments
        return original_completion(*args, **kwargs)


    # Replace the original function with our patched version
    litellm.completion = patched_completion
    print("litellm.completion successfully patched")

    print(f"Making patched litellm call with model={MODEL_ID}")
    response = litellm.completion(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Say 'Hello from patched litellm!'"}],
        temperature=0.2,
        max_tokens=30
    )

    response_text = response.choices[0].message.content
    print(f"✅ Success! Response: {response_text}")
except Exception as e:
    print(f"❌ Error with patched litellm: {e}")

print("\nTests completed. Please review the results to determine which approach works for your Cloudera ML deployment.")