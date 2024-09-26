import matplotlib.pyplot as plt
import numpy as np

values_inception = [90.0, 96.49]
values_dense = [87.0, 96.75]
values_xception = [86.50, 92.75]

labels = ['Nucleus Images', 'Phase Images']

x = np.arange(len(labels))
width = 0.2  

fig, ax = plt.subplots(figsize=(10,6))

rects1 = ax.bar(x - width, values_inception, width, label='InceptionV3', color='blue')
rects2 = ax.bar(x, values_dense, width, label='DenseNet121', color='red')
rects3 = ax.bar(x + width, values_xception, width, label='Xception', color='green')

ax.set_ylabel('Mean Accuracy in Percent')
ax.set_xlabel('\nPerformance of CNN architectures on Different Image Modalities')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.subplots_adjust(bottom=0.25)  
plt.title('Performance of CNN Architectures on Different Modalities')

plt.show()