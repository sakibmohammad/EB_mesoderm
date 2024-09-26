import matplotlib.pyplot as plt
import numpy as np

values_unet = [58.08, 67.97]
values_Aunet = [61.35, 69.31]


labels = ['Nucleus Images', 'Phase Images']

x = np.arange(len(labels))
width = 0.2 

# Adjust the figure size
fig, ax = plt.subplots(figsize=(10,6))

rects1 = ax.bar(x - width, values_unet, width, label='Unet', color='blue')
rects2 = ax.bar(x, values_Aunet, width, label='Attention Unet', color='red')

ax.set_ylabel('Mean IoU in Percent')
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

plt.subplots_adjust(bottom=0.25)  # More space for the legend
plt.title('Performance of CNN Architectures on Different Modalities')

plt.show()

