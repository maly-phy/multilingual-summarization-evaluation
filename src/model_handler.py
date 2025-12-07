import time, random


class ModelHandler:
    def __init__(self, client, model_name, max_tokens):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens

    def base_call_model(self, messages):
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=None,
            n=1,
        )
        return response

    def call_model(self, system_prompt, user_prompt, max_attempts=3, base_delay=3.0):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(max_attempts):
            try:
                response = self.base_call_model(messages=messages)
                response = response.choices[0].message.content.strip()
                return response

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {str(e)}\n")

                if "429" in str(e):
                    sleep_time = base_delay * (2**attempt + 1) + (
                        random.randint(0, 1000) / 1000
                    )
                    print(
                        f"Rate limit exceeded. Retrying in {sleep_time/60:.2f} minutes..."
                    )
                    time.sleep(sleep_time)
                    attempt += 1
                else:
                    print(f"Error calling model: {str(e)}", flush=True)
                    break

            finally:
                time.sleep(base_delay)
