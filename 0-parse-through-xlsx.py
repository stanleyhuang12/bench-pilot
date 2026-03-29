import pandas as pd 

df = pd.read_csv('benchmark_submission.csv')

column_maps = {
    "Timestamp": "timestamp", 
    "Email Address": "email", 
    'Define the construct: What are you attempting to measure? Is there a literature on this construct that you can point to?': 'construct', 
    'Relate the construct: Conceptually, how does this benchmark relate to other existing constructs, especially those that are already represented in this project?': "construct_relate", 
    'Justify the construct: Please provide a narrative description, including citations, for why this LLM behavior that is being benchmarked, relates to the dimension(s) of human flourishing you identified above.': "construct_justify",
    'User demographic(s): What demographic(s) of interest would you be interested in testing? We may simulate conversations that reflect this demographic. We may provide benchmarks where the model explicitly knows the demographics or when the model has to infer these demographics from previous conversations.': "demographics",
    'User context(s) - known: Define what the model knows about the user from previous interactions. What has the user previously expressed that may elicit the behavior of interest. ': "user_context", 
    'User context(s) - unknown: Define characteristics of the user that the model cannot directly observe. We may simulate conversations that reflect this context without providing this context to the model.': "implicit_context", 
    'User message(s): Provide one or more messages that are designed to elicit the behavior of interest.': "user_message", 
    'LLM-as judge prompt: Provide one or more LLM prompts that allows for a yes/no judgment as to whether a behavior is present in any LLM response. If you provide more than one response, we may test various versions.': "llm_as_judge_prompt", 
    'Positively Scored Examples: Provide one or more specific examples of this behavior being present in a response that we can use to validate the LLM-as-judge prompt.': "pos_examples", 
    'Negatively Scored Examples: Provide one or more specific examples of this behavior being absent in a response that we can use to validate the LLM-as-judge prompt.': "neg_examples", 
    'Name the Construct: Provide a name for your proposed benchmark idea (Name/Concept).  ': "construct_name", 
}
cols_of_interest = [v for _, v in column_maps.items()]
df = df.rename(columns=column_maps)
df = df[cols_of_interest]





print(df[cols_of_interest].columns)