import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from resources.model_utils import evaluate_model
from resources.utils import label_distribution, prediction_standardized, aggregate_generator_labels
from resources.model_utils_pt import Model


class TrainingSaver:
    def __init__(self, path_out, nb_classes, df_train, df_valid, df_test, column_label, stopped_epoch, model_path, prefix):
        self.path_out = path_out
        self.nb_classes = nb_classes
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.column_label = column_label
        self.stopped_epoch = stopped_epoch

        # build model
        model = Model()
        # load check point
        model.load_state_dict(torch.load(save_path))

        self.base_path = "{}class_{}_trainsize_{}".format(path_out, str(nb_classes), str(len(df_train)))
        self.base_path = self.base_path + prefix

    def save_class_count(self):
        """Save class counts"""
        train_path = self.base_path + "_classcount_train.csv"
        valid_path = self.base_path + "_classcount_val.csv"
        test_path = self.base_path + "_classcount_test.csv"

        class_count_train = label_distribution(data_frame=self.df_train, column_target=self.column_label)
        class_count_val = label_distribution(data_frame=self.df_valid, column_target=self.column_label)
        class_count_test = label_distribution(data_frame=self.df_test, column_target=self.column_label)

        np.savetxt(train_path, class_count_train, delimiter=',', fmt='%d', header="Class,Count")
        np.savetxt(valid_path, class_count_val, delimiter=',', fmt='%d', header="Class,Count")
        np.savetxt(test_path, class_count_test, delimiter=',', fmt='%d', header="Class,Count")

        print("DONE: Saving class counts.")

    def save_stopped_epoch(self):
        path = self.base_path + "_stopped_epoch.txt"
        np.savetxt(path, self.stopped_epoch, delimiter=',', fmt='%d')

        print("DONE: Saving stopped epoch value.")

    def save_accuracy(self):
        # Prediction
        train_predictions, train_acc = self.generate_predictions(self.df_train)
        valid_predictions,  valid_acc = self.generate_predictions(self.df_valid)
        test_predictions,  test_acc = self.generate_predictions(self.df_test)

        train_true_labels = aggregate_generator_labels(data_generator=train_generator)
        valid_true_labels = aggregate_generator_labels(data_generator=valid_generator)
        test_true_labels = aggregate_generator_labels(data_generator=test_generator)

        accuracy_list = [acc_train, acc_valid, acc_test]


        # Save accuracy
        filename = self.base_path + "_accuracy.txt"
        np.savetxt(filename, accuracy_list, delimiter=',', fmt='%f', header="Train,Valid,Test")

        print("DONE: Model evaluation.")

    @torch.no_grad()
    def generate_predictions(loader):
        model = self.model
        total_correct = 0.0
        total_datapoints = 0.0
        all_preds = torch.tensor([])
        for X, y in test_loader:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            outputs = model(X)
            predictions = torch.argmax(outputs, dim = 1)
            all_preds = torch.cat(
                (all_preds, preds)
                ,dim=0
            )
            matches = predictions == y
            total_correct += len(matches.count(True))
            total_datapoints += len(matches)
        acc = total_correct / total_datapoints
        return all_preds, acc
