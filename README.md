# Med-VQA
The implementation of **Medical Visual Question Answering via Conditional Reasoning** [ACM MM 2020]  
We evaluate our proposal on [VQA-RAD dataset](https://www.nature.com/articles/sdata2018251#data-citations).

 

## Conditional Reasoning Framework  

We propose QCR [**Question-Conditioned Reasoning Module**] and TCR [**Type-Conditioned Reasoning**] modules, which guide the importance selection over multimodal fusion features and learn different reasoning skills for different types of tasks separately. The detailed architecture is shown in the figure below.  
![Image text](https://raw.githubusercontent.com/Awenbocc/Med-VQA/master/pics/architecture.png)

## Results

|  | Overall | Open-ended | Closed-ended |
| :----:| :----: | :----: | :----: |
| Base | 66.1 | 49.2 | 77.2 | 
| Our proposal | 71.6 | 60.0 | 79.3|

Our proposal achieves significantly increased accuracy in predicting answers to both closed-ended and open-ended questions, especially for open-ended questions.

## Training

### 1.Dependence
```
pip install -r requirements.txt
```

### 2.Preparing
All pre-prepared data can be found in ```data``` package
