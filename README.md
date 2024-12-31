# Project Name

This is an example of how to include Python code in a Markdown file for GitHub.

## Example Python Code
```python
vc = train['Basic_Demos-Enroll_Season'].value_counts()

# Map labels to seasons
season_map = {0: 'Spring', 1: 'Summer', 2: 'Fall', 3: 'Winter'}

# Create labels using the map
labels = [season_map[label] for label in vc.index]

# Plot the pie chart with the updated labels
plt.pie(vc, labels=labels)
# Plot the pie chart
plt.pie(vc.values, labels=labels, autopct="%1.1f%%")  # Add percentages to pie slices
plt.title('Season of Enrollment')
plt.show()
```

![image](https://github.com/user-attachments/assets/9705c7ae-cd3b-4777-bead-9eb636edb0c0)

Saved to submission.csv
tyyhf
