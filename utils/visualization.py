from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def print_shape(a,b):
    """
    Function that prints the shape of the numpy arrays passed as arguments
    """
    print("Size of Training Samples")
    print("="*30)
    print(a.shape)
    print("Size of Testing Samples")
    print("="*30)
    print(b.shape)

def print_bal_class(tag,Y):
   print(f'{tag}:{Y.value_counts()}')

def print_average_word_length(tag, X):
    len_acc = 0
    for text in X:
        len_acc+=len(text.split(" "))
    print(f'{tag}:{len_acc/len(X)}')

def print_average_char_length(tag, X):
    len_acc = 0
    for text in X:
        dx = text.split(" ")
        for rev in dx:
            len_acc+=len(rev)
    print(f'{tag}:{len_acc/len(X)}')

def print_metrics(y_test,y_pred):
    pscore = precision_score(y_test, y_pred)*100
    rscore = recall_score(y_test, y_pred)*100
    ascore = accuracy_score(y_test, y_pred)*100
    fscore = f1_score(y_test, y_pred)*100
    print('Precision: %.3f' % pscore)
    print('Recall: %.3f' % rscore)
    print('Accuracy: %.3f' % ascore)
    print('F1 Score: %.3f' % fscore)
    return [pscore,rscore,ascore,fscore]

def print_df(df):
    print(f"Dataframe Shape: {df.shape}")
    print(f"Positive Class Shape: {df[df['sentiment']==1].shape}")
    print(f"Negative Class Shape: {df[df['sentiment']==0].shape}")