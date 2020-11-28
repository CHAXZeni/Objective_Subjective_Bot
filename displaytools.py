import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------------------------------------------------------------------------#

# Plots the training/validation accuracy and loss graphs. Gives final numbers for the loss and accuracy for each set as well as run time.
def Display_Results(Train_Loss, Train_Accuracy, Valid_Loss, Valid_Accuracy, time_taken, test_accuracy, test_loss):
    
    # Creating the Epochs
    X = np.arange(0,len(Train_Accuracy),1)
    Y = np.arange(0,len(Train_Accuracy),len(Train_Accuracy)/len(Valid_Accuracy))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout(pad=3.0)
    
    #Plot Loss
    ax1.plot(X, Train_Loss, 'r', label='Training')
    ax1.plot(Y, Valid_Loss , 'b', label='Validation')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    #Plot Accuracy
    ax2.plot(X, Train_Accuracy, 'r', label='Training')
    ax2.plot(Y, Valid_Accuracy , 'b', label='Validation')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.show()
    
    print('Final Training Accuracy:', Train_Accuracy[len(Train_Accuracy)-1])
    print('Final Training Loss:', Train_Loss[len(Train_Loss)-1])
    print('Final Validation Accuracy:', Valid_Accuracy[len(Valid_Accuracy)-1])
    print('Final Training Loss:', Valid_Loss[len(Valid_Loss)-1])
    print('Final Test Accuracy:', test_accuracy)
    print('Final Test Loss:', test_loss)
    print('Time to Train:', time_taken)
