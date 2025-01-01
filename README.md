# 🏆📖*Predicting Problematic Internet Usage Based on Physical Activity*

Welcome to our GitHub repository for the "Predicting Problematic Internet Usage" competition! This project aims to address a growing concern in today’s digital age—problematic internet use among children and adolescents. By leveraging physical activity and fitness data, we strive to build a predictive model capable of identifying early signs of problematic internet behavior.

## Competition Overview

The goal of this competition is to analyze children’s physical activity data and predict their level of problematic internet usage. This initiative is critical in promoting early interventions that encourage healthier digital habits and prevent mental health issues such as anxiety and depression.

## Why This Matters

Traditional methods of measuring problematic internet use often involve complex clinical assessments, which can be inaccessible due to cultural, linguistic, or logistical barriers. On the other hand, physical activity and fitness data are widely available and require minimal intervention, making them ideal proxies for identifying problematic internet use. Excessive technology use often manifests through changes in physical behavior, such as reduced activity levels and poorer posture. By harnessing these indicators, we aim to create scalable solutions to this growing problem.

## Competition Details

- **Start Date:** September 19, 2024
- **Entry Deadline:** December 12, 2024
- **Team Merger Deadline:** December 12, 2024
- **Final Submission Deadline:** December 19, 2024

### Prizes
- **1st Place:** $15,000
- **2nd Place:** $10,000
- **3rd Place:** $8,000
- **4th - 8th Places:** $5,000 each

## Data Source

This competition utilizes data provided by the **Healthy Brain Network**, a mental health study conducted by the Child Mind Institute. This initiative is supported by the California Department of Health Care Services and sponsors like Dell Technologies and NVIDIA. The dataset includes valuable indicators such as:
- Accelerometer measures (e.g., X, Y, Z acceleration, and derived metrics like ENMO)
- Light exposure and time-of-day patterns
- Demographic and contextual features

## Acknowledgments

Special thanks to our sponsors, the **Child Mind Institute**, **Dell Technologies**, and **NVIDIA**, for their support in making this competition possible. By contributing to this challenge, you are helping pave the way for healthier digital habits and a brighter future for children and adolescents worldwide.


# 📊🔍*Exploratory Data Analysis*
<p align="center">
  <img src="https://github.com/user-attachments/assets/7f56ae4f-ca94-4a85-a51f-7eeb20c88d7a">
</p>

**It appears that the enrollment is nearly the same across all seasons. After testing, we came to the conclusion that it is better to not use season specific columns.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/6a8b4bc1-979d-4dd7-a68d-601fbd1a8662">
</p>

**In our dataset, the majority of individuals use the internet for less than an hour, while those using it for more than 3 hours are quite few.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/020cc65e-b6a7-481a-b308-f3bc73455d59">
</p>

**The pie chart above clearly shows that only 9.9% of people use the internet excessively, while the rest of the users either spend around 2 hours or less online.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/92eab349-809b-4651-ad63-61fbaae5d9b0">
</p>

**The pie chart clearly shows that the majority of people, 58.3%, have no problems, while 26.7% experience only mild issues. This means that 85% of individuals in our dataset have either no problems or mild ones. In contrast, 15% of the population falls into more serious categories, with 13.8% having moderate problems and only 1.2% displaying severe issues. Therefore, in our dataset, only 1.2% of people have a severe problem, which is a very small percentage compared to the other three groups.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/801531fa-da07-4acd-aa48-9f4997acd1b6">
</p>

**Here, we have a scatter plot for our target variable, SII, and the number of hours of internet usage. While the plot is interesting, it is also somewhat perplexing. It shows that all possible SII values are present across all ranges of internet usage hours per day, making it difficult to discern how the SII value changes with varying internet usage time. A potential solution to better understand this relationship would be to include a count of computer/internet users along with the SII and internet usage hours per day. This could help clarify the distribution and trend.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/c98d11c5-1112-4b82-8573-50778307f71e">
</p>

**In the above chart, we have the SII, PreInt_EduHx-computerinternet_hoursday, and the count of PreInt_EduHx-computerinternet_hoursday. These three variables together provide a more comprehensive understanding of the data compared to the previous chart, which only used two features. By incorporating the count, we can better visualize and interpret the relationship between internet usage hours, educational history, and the severity of impairment.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/6a5e95dd-7353-44bc-ac34-5b2753dd96bc">
</p>

**The majority of people in this dataset are male, although there is also a significant number of females represented.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/e79bca87-a373-4b64-bd1d-0da20fa17515">
</p>

**The majority of people in our dataset are young children or teenagers, with no elderly individuals present. It's important to note that we are tasked with determining the Severity Impairment Index (SII) for people in the future.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/48419bb8-ac72-4462-a4f5-4c04ecce41f2">
</p>

**The scatter plot above shows that the Sleep Disturbance Scale values for individuals with SII scores of 0, 1, and 2 fall within a similar range, with only a few outliers. For SII 3, we observe fewer entries, which can explain the smaller number of marks/dots in that section of the plot. However, even though there are fewer dots for SII 3, they still lie within the same range as those for SII 0, 1, and 2. The range for SII 3 is smaller compared to SII 0, 1, and 2, but I refer to it as the "same range" because there aren't a significant number of points outside the range of the other SII scores. A surprising finding is that for SII 3, we don't observe very high or very low sleep disturbance values. Ideally, we would expect SII 3 to correspond to higher sleep disturbance scores, but this isn't reflected in the data.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/a1efdbec-6298-48e6-836e-9ad1416c8139">
</p>

**The violin plot illustrates the distribution of data points, showing that the majority of points fall within the sleep disturbance scale range of 30 to 50. This distribution appears quite similar for SII values of 0, 1, and 2. However, for SII value 3, the data points seem slightly fewer compared to those for SII values 0, 1, and 2.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/ca32c59e-1689-44ea-8b8d-8a57c5aecbdb">
</p>

**The recorded values for BIA BMI and Physical BMI appear to be slightly different. There are two possible reasons for this:**

**1. There may be some discrepancy or error in the measurement of BMI.**

**2. The two BMIs could have been calculated at different ages for the same individual.**
