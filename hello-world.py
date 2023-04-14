import semantic_kernel as sk
from semantic_kernel.ai.open_ai import OpenAITextCompletion, AzureTextCompletion

kernel = sk.create_kernel()

# Prepare OpenAI backend using credentials stored in the `.env` file
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.config.add_text_backend("dv", OpenAITextCompletion("text-davinci-003", api_key, org_id))

# Alternative using Azure:
# deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
# kernel.config.add_text_backend("dv", AzureTextCompletion(deployment, endpoint, api_key))

# Wrap your prompt in a function
prompt = kernel.create_semantic_function("""
1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.

Give me the TLDR in exactly 5 words.""")

# Run your prompt
print(prompt()) # => Robots must not harm humans.