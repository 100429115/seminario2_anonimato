from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import statistics

def performance_rates(y_test, y_pred):

    # it return the performance rates, f1, accuracy, TFP and TFN
    f1 = statistics.mean(f1_score(y_test, y_pred, average=None))
    accuracy = accuracy_score(y_test, y_pred)

    matrix = confusion_matrix(y_test, y_pred)
    n_users = n_columns = matrix.shape[1]

    fp_total = 0
    fn_total = 0

    # u represents each of the users
    for u in range(n_users):

        # false positives
        fp = 0

        # true negatives
        tn = 0

        # false negatives
        fn = 0

        # true positives
        tp = 0

        for r in range(matrix.shape[0]):
            for c in range(n_columns):
                # For clarity, each metric is calculated separately
                if r == u and c == u:
                    tp = matrix[r][c]
                if r != u and c != u:
                    tn = tn + matrix[r][c]
                if r == u and c != u:
                    fp = fp + matrix[r][c]
                if r != u and c == u:
                    fn = fn + matrix[r][c]

        if fp == 0 and tn == 0:
            user_fp_rate = 0

        else:
            user_fp_rate = fp / (fp + tn)

        fp_total += user_fp_rate

        if fn == 0 and tp == 0:
            user_fn_rate = 0

        else:
            user_fn_rate = fn / (fn + tp)

        fn_total += user_fn_rate

    mean_fp = fp_total / n_users
    mean_fn = fn_total / n_users

    return f1, accuracy, mean_fp, mean_fn