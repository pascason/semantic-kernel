import semantic_kernel as sk
from semantic_kernel.ai.open_ai import OpenAITextCompletion, AzureTextCompletion

kernel = sk.create_kernel()

# Prepare OpenAI backend using credentials stored in the `.env` file
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.config.add_text_backend("dv", OpenAITextCompletion("text-davinci-003", api_key, org_id))


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


prompt = """{{$input}}
Summarize the content above.
"""

summarize = kernel.create_semantic_function(prompt, max_tokens=2000, temperature=0.2, top_p=0.5)

# note: using skills from the samples folder
skills_directory = "samples/skills"

#funFunctions = kernel.import_semantic_skill_from_directory(skills_directory, "FunSkill")
#FunFunction = funFunctions["Joke"]
#result = FunFunction("A Neapolitan nerd walks into a bar")
#print(result)

analyticFunctions = kernel.import_semantic_skill_from_directory(skills_directory, "AnalyticSkill")
AnomalyFunction = analyticFunctions["FindTimelineAnomalies"]
result = AnomalyFunction("{start = '1/1/2023', end = '1/30/2023', data {2976,2992,2160,2144,2524,3088,2000,2508,2184,3040,2080,2580,2564,3152,2656,3040,2016,2256,2028,2064,2736,2856,2804,3196,2992,2072,2464,2520,2620,3675}}")

print(result)
