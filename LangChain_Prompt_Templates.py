#!/usr/bin/env python
# coding: utf-8

# # Mastering Prompt Templates in LangChain
# 
# **Assignment: Mini Prompt Engine — GenAI Prompt Engineering (LangChain Prompt Templates)**
# 
# This notebook builds *dynamic, reusable* prompt systems with LangChain instead of
# hardcoded prompts. It covers all seven assignment tasks.
# 
# **Rules followed throughout:**
# - No f-strings for prompt construction
# - No hardcoding of prompt text inside logic
# - Every template declares its `input_variables`
# - Logic is kept separate from templates (modular, reusable design)
# 
# **Pipeline:** `User Input → Validation → Prompt Template → Dynamic Prompt Generation → Output`
# 

# ## Setup
# 
# Install LangChain (the core package provides `PromptTemplate` and
# `ChatPromptTemplate`). Run once per environment.

# In[1]:


# Install LangChain (quiet). Safe to re-run.
# In Google Colab / a fresh environment, uncomment the next line:
# get_ipython().system('pip install -q langchain langchain-core')   # (notebook-only; install via: pip install langchain langchain-core)
print("If LangChain is not installed, uncomment the pip line above and run this cell.")


# In[2]:


# Core imports. In modern LangChain (v0.1+ / v1.x) the prompt classes live in
# langchain_core.prompts. We fall back to the classic langchain.prompts path for
# older installs so the notebook runs on either version.
try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
except ImportError:  # older LangChain (<0.1)
    from langchain.prompts import PromptTemplate, ChatPromptTemplate

print("LangChain prompt classes imported successfully.")


# ## Task 1: Replace Hardcoded Prompts (10%)
# 
# **Original (hardcoded with an f-string):**
# 
# ```python
# def explain_topic(topic):
#     return f"Explain {topic} in simple terms for beginners"
# ```
# 
# We replace this with a reusable `PromptTemplate`. The template text is declared
# once, `topic` is the only `input_variable`, and the same object can be reused for
# any topic.

# In[3]:


# Reusable template — declared once, no f-strings.
explain_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms for beginners",
)

def explain_topic(topic):
    """Return a beginner-friendly explanation prompt for any topic."""
    # .format() substitutes the input_variable; logic stays separate from text.
    return explain_template.format(topic=topic)

# Demonstrate reusability with a few topics.
for t in ["Recursion", "Blockchain", "Photosynthesis"]:
    print(explain_topic(t))


# ## Task 2: Multi-Input Prompt System (15%)
# 
# A single template driven by three inputs: `topic`, `audience`, and `tone`.
# 
# **Template:** `Explain {topic} for {audience} in a {tone} tone`

# In[4]:


# One template, three input_variables.
multi_input_template = PromptTemplate(
    input_variables=["topic", "audience", "tone"],
    template="Explain {topic} for {audience} in a {tone} tone",
)

def build_multi_prompt(topic, audience, tone):
    """Format the multi-input template. Logic is separate from the template."""
    return multi_input_template.format(topic=topic, audience=audience, tone=tone)

# Test cases from the assignment: topic | audience | tone
test_cases = [
    ("AI", "beginners", "friendly"),
    ("Python", "kids", "fun"),
    ("Deep Learning", "engineers", "technical"),
]

for topic, audience, tone in test_cases:
    print(build_multi_prompt(topic, audience, tone))


# ## Task 3: Prompt Variations Engine (15%)
# 
# Three distinct `PromptTemplate`s sharing the same input variable `topic`:
# 
# - **Teaching** → Explain {topic} clearly step by step
# - **Interview** → Ask 3 questions about {topic}
# - **Storytelling** → Explain {topic} as a story
# 
# Run all three for `topic = "Machine Learning"`.

# In[5]:


# A dictionary of named templates keeps the variations modular and reusable.
variation_templates = {
    "Teaching": PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} clearly step by step",
    ),
    "Interview": PromptTemplate(
        input_variables=["topic"],
        template="Ask 3 questions about {topic}",
    ),
    "Storytelling": PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} as a story",
    ),
}

def run_variations(topic):
    """Generate every prompt variation for a given topic."""
    return {name: tmpl.format(topic=topic) for name, tmpl in variation_templates.items()}

topic = "Machine Learning"
for style, prompt in run_variations(topic).items():
    print(style + " -> " + prompt)


# ## Task 4: ChatPromptTemplate System (15%)
# 
# A chat-based prompt system with two roles:
# 
# - **System** → defines behavior (driven by the chosen role)
# - **User** → dynamic input (the topic)
# 
# Supported roles: `teacher`, `interviewer`, `motivator`.

# In[6]:


# System behaviour per role — declared as data, not hardcoded in logic.
role_behaviors = {
    "teacher": "You are a patient teacher who explains concepts clearly and step by step.",
    "interviewer": "You are a technical interviewer who asks probing, thoughtful questions.",
    "motivator": "You are an energetic motivator who inspires and encourages the learner.",
}

# ChatPromptTemplate with a system message (role behaviour) and a user message (topic).
chat_template = ChatPromptTemplate.from_messages([
    ("system", "{behavior}"),
    ("user", "Let's talk about {topic}."),
])

def build_chat_prompt(role, topic):
    """Build chat messages for a role + topic. Falls back to 'teacher' if unknown."""
    behavior = role_behaviors.get(role, role_behaviors["teacher"])
    messages = chat_template.format_messages(behavior=behavior, topic=topic)
    return messages

# Example from the assignment.
example_messages = build_chat_prompt(role="teacher", topic="Neural Networks")
for m in example_messages:
    print(m.type.upper() + ": " + m.content)

print("\n--- All roles for topic 'Neural Networks' ---")
for r in role_behaviors:
    print("\nROLE =", r)
    for m in build_chat_prompt(r, "Neural Networks"):
        print("  " + m.type.upper() + ": " + m.content)


# ## Task 5: Input Validation Layer (10%)
# 
# `validate_inputs(audience, tone)` enforces:
# 
# - `audience ∈ [beginner, intermediate, expert]`
# - `tone ∈ [formal, casual, fun]`
# 
# If a value is invalid we **assign a default** (and warn) rather than crash, so the
# pipeline keeps flowing. A strict raising version is also shown.

# In[7]:


# Allowed values declared once as configuration.
VALID_AUDIENCES = ["beginner", "intermediate", "expert"]
VALID_TONES = ["formal", "casual", "fun"]

DEFAULT_AUDIENCE = "beginner"
DEFAULT_TONE = "casual"

def validate_inputs(audience, tone):
    """Validate audience and tone. Invalid values fall back to defaults (with a warning).
    Returns a cleaned (audience, tone) tuple."""
    if audience not in VALID_AUDIENCES:
        print("Warning: invalid audience '" + str(audience) +
              "'. Using default '" + DEFAULT_AUDIENCE + "'.")
        audience = DEFAULT_AUDIENCE
    if tone not in VALID_TONES:
        print("Warning: invalid tone '" + str(tone) +
              "'. Using default '" + DEFAULT_TONE + "'.")
        tone = DEFAULT_TONE
    return audience, tone

def validate_inputs_strict(audience, tone):
    """Strict variant: raise ValueError on invalid input."""
    if audience not in VALID_AUDIENCES:
        raise ValueError("audience must be one of " + str(VALID_AUDIENCES))
    if tone not in VALID_TONES:
        raise ValueError("tone must be one of " + str(VALID_TONES))
    return audience, tone

# Valid input passes through unchanged.
print(validate_inputs("expert", "formal"))

# Invalid input falls back to defaults.
print(validate_inputs("scientist", "angry"))

# Strict variant raises.
try:
    validate_inputs_strict("scientist", "angry")
except ValueError as e:
    print("Raised ValueError:", e)


# ## Task 6: Prompt Generator App (15%)
# 
# `generate_prompt(topic, audience, tone, style)` ties everything together:
# validation → template selection → dynamic prompt generation.
# 
# Supported styles: `teaching`, `interview`, `storytelling`.
# 
# **Example output:** `Explain Neural Networks for beginners in a fun storytelling style`

# In[8]:


# One master template covering all styles via input_variables — no hardcoding.
generator_template = PromptTemplate(
    input_variables=["topic", "audience", "tone", "style"],
    template="Explain {topic} for {audience} in a {tone} {style} style",
)

VALID_STYLES = ["teaching", "interview", "storytelling"]
DEFAULT_STYLE = "teaching"

def generate_prompt(topic, audience, tone, style):
    """Full pipeline: validate inputs, check style, then build the prompt."""
    # Step 1: validate audience & tone (reuses Task 5).
    audience, tone = validate_inputs(audience, tone)

    # Step 2: validate style.
    if style not in VALID_STYLES:
        print("Warning: invalid style '" + str(style) +
              "'. Using default '" + DEFAULT_STYLE + "'.")
        style = DEFAULT_STYLE

    # Step 3: generate the dynamic prompt from the template.
    return generator_template.format(topic=topic, audience=audience, tone=tone, style=style)

# Assignment example.
print(generate_prompt("Neural Networks", "beginner", "fun", "storytelling"))

# A few more, including an invalid combination that gets defaulted.
print(generate_prompt("Databases", "expert", "formal", "teaching"))
print(generate_prompt("Robotics", "guru", "spicy", "interview"))


# ## Task 7: Template Reusability Test (10%)
# 
# Use **one** template and run it with **5 different** input sets. Same structure,
# different outputs — proving the template is reusable.

# In[9]:


# Reuse the single multi-input template from Task 2.
reusable_template = multi_input_template

five_inputs = [
    {"topic": "AI", "audience": "beginners", "tone": "friendly"},
    {"topic": "Quantum Computing", "audience": "engineers", "tone": "technical"},
    {"topic": "Cooking", "audience": "kids", "tone": "fun"},
    {"topic": "Economics", "audience": "students", "tone": "formal"},
    {"topic": "Music Theory", "audience": "hobbyists", "tone": "casual"},
]

for i, inputs in enumerate(five_inputs, start=1):
    # Same template object reused for every input set.
    print(str(i) + ". " + reusable_template.format(**inputs))


# ## End-to-End Pipeline Demo
# 
# A final demonstration of the full flow:
# `User Input → Validation → Prompt Template → Dynamic Prompt → Output`.

# In[10]:


def prompt_engine(topic, audience, tone, style):
    """Mini Prompt Engine end-to-end."""
    print("User input  :", {"topic": topic, "audience": audience, "tone": tone, "style": style})
    prompt = generate_prompt(topic, audience, tone, style)  # validation happens inside
    print("Final prompt:", prompt)
    return prompt

_ = prompt_engine("Machine Learning", "beginner", "fun", "storytelling")


# ## Summary
# 
# | Task | Concept | Status |
# |------|---------|--------|
# | 1 | Replace hardcoded prompt with `PromptTemplate` | Done |
# | 2 | Multi-input template (topic, audience, tone) | Done |
# | 3 | Prompt variations engine (3 templates) | Done |
# | 4 | `ChatPromptTemplate` with system/user roles | Done |
# | 5 | Input validation layer | Done |
# | 6 | Prompt generator app | Done |
# | 7 | Template reusability test (1 template, 5 inputs) | Done |
# 
# This assignment shifts the focus from *writing prompts* to *designing reusable
# prompt systems* — a core skill for real-world Generative AI applications.
