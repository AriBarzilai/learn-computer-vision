# %% [markdown]
# # EX 1.1- basic python: Pyramid case
# Implement a function that get a string input and outputs the same word, only each odd char is lower
# case and each even letter is upper case
# You can assume that the input is a valid string which contains only english letters.
# %%


def pyramid_case(in_word):
    out_word = ''
    for i, c in enumerate(in_word):
        if i%2 == 0: # i am assuming 0-index for even/odd
            out_word += c.upper()
        else:
            out_word += c.lower()
    return out_word
# %%


def pyramid_case_one_liner(in_word):
    return ''.join([char.upper() if i % 2 == 0 else char.lower() for i, char in enumerate(in_word)])


# %%
# test functions here
input_words = ["hello", "world", "", "I", "am", "LEARNING", "Python"]

print("==== pyramid_case() results:")
for word in input_words:
    print(pyramid_case(word))

print("\n==== pyramid_case_one_liner() results:")
for word in input_words:
    print(pyramid_case_one_liner(word))


# %%
