# name = "Asha"
# years = 4
# skills = ["python", "sql", "aws"]  # list  ~ TS array
# profile = {"name": name, "years": years}  # dict  ~ TS object / Map
# point = (12.9, 77.6)  # tuple ~ readonly array, cannot change
# unique = {"python", "python", "go"}  # set   ~ TS Set -> {'python', 'go'}


# print(f"{name} has {years} and {len(skills)} skills")

# if years > 5 and "python" in skills:
#     level = "senior"
# elif years > 2:
#     level = "mid"
# else:
#     level = "junior"

# print(level)

# for i, skill in enumerate(skills, start=1):
#     print(i,skill)


# upper = [s.upper() for s in skills]
# print(upper)

# short = [s for s in skills if len(s) <= 3]
# print(short)


# def greet(name: str, greeting: str = "Hello"):
#     return f"{greeting} {name}"

# print(greet("Shadab", "Goog morning "))


# manager = None
# if manager is None:
#     print("no manager yet")


# # Falsy values: None, 0, "", [], {}, set()  (an empty list is falsy, unlike in JS!)
# if not []:
#     print("empty list is falsy")


def count_skills(resumes: list[list[str]]):
    """Count how many resumes mention each skill (case-insensitive, lowercase keys).
    count_skills([["Python", "SQL"], ["python"]]) -> {"python": 2, "sql": 1}
    """
    # TODO: your code here
    dict = {}
    for skills in resumes:
        for skill in skills:
            skill = skill.lower()
            if skill in dict:
                dict[skill] += 1
            else: 
                dict[skill] = 1
    return dict

list = count_skills([["Python", "SQL"], ["python"]])
print(list)

if __name__ == "__main__":
    print("Direct")