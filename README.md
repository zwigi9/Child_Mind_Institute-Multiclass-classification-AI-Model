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

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Basic_Demos-Enroll_Season</th>
      <th>Basic_Demos-Age</th>
      <th>Basic_Demos-Sex</th>
      <th>CGAS-Season</th>
      <th>CGAS-CGAS_Score</th>
      <th>Physical-Season</th>
      <th>Physical-BMI</th>
      <th>Physical-Height</th>
      <th>Physical-Weight</th>
      <th>...</th>
      <th>PCIAT-PCIAT_18</th>
      <th>PCIAT-PCIAT_19</th>
      <th>PCIAT-PCIAT_20</th>
      <th>PCIAT-PCIAT_Total</th>
      <th>SDS-Season</th>
      <th>SDS-SDS_Total_Raw</th>
      <th>SDS-SDS_Total_T</th>
      <th>PreInt_EduHx-Season</th>
      <th>PreInt_EduHx-computerinternet_hoursday</th>
      <th>sii</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00008ff9</td>
      <td>Fall</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>Winter</td>
      <td>51.0</td>
      <td>Fall</td>
      <td>16.877316</td>
      <td>46.0</td>
      <td>50.8</td>
      <td>...</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>55.0</td>
      <td>Spring</td>
      <td>39.0</td>
      <td>55.0</td>
      <td>Fall</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000fd460</td>
      <td>Summer</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>Spring</td>
      <td>65.0</td>
      <td>Fall</td>
      <td>14.035590</td>
      <td>48.0</td>
      <td>46.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Fall</td>
      <td>46.0</td>
      <td>64.0</td>
      <td>Summer</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00105258</td>
      <td>Summer</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>Fall</td>
      <td>71.0</td>
      <td>Fall</td>
      <td>16.648696</td>
      <td>56.5</td>
      <td>75.6</td>
      <td>...</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>Fall</td>
      <td>38.0</td>
      <td>54.0</td>
      <td>Summer</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00115b9f</td>
      <td>Winter</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>Fall</td>
      <td>71.0</td>
      <td>Summer</td>
      <td>18.292347</td>
      <td>56.0</td>
      <td>81.6</td>
      <td>...</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>44.0</td>
      <td>Summer</td>
      <td>31.0</td>
      <td>45.0</td>
      <td>Winter</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0016bb22</td>
      <td>Spring</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>Summer</td>
      <td>65.0</td>
      <td>Spring</td>
      <td>17.937682</td>
      <td>55.0</td>
      <td>77.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>Spring</td>
      <td>39.0</td>
      <td>55.0</td>
      <td>Spring</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>

```HTML
	<iframe src="https://www.kaggle.com/embed/taimour/sparcepca-yeo-jhonson-eda-cmi-piu?cellIds=10&kernelSessionId=209757407" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="💻 SparcePCA Yeo-Jhonson 📊 EDA | CMI PIU"></iframe>
```
