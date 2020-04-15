class MultiNB:
    def fit(self, X_train, label):
        num_of_document = len(label)
        dim_fea_vector = len(X_train[0])
        setlabel = list(set(label))
        num_of_labels = len(setlabel)
        numdoc_of_each_class = dict.fromkeys(setlabel, 0)
        num_words_in_class = []
        for i in setlabel:
            num_words_in_class.append(np.zeros(dim_fea_vector))
        for i in range(num_of_document):
            numdoc_of_each_class[label[i]] += 1
            num_words_in_class[setlabel.index(label[i])] += X_train[i]

        for i in range(num_of_labels):
            total_words = sum(num_words_in_class[i])+dim_fea_vector
            num_words_in_class[i] = (num_words_in_class[i]+1)/total_words
            num_words_in_class[i] = np.log10(num_words_in_class[i])

        log_pc = np.array(list(numdoc_of_each_class.values()))
        log_pc = np.log10(log_pc/num_of_document)
        self.log_lamda_class = num_words_in_class
        self.log_pc = log_pc
        self.dim_fea_vector = dim_fea_vector
        self.labels = setlabel
        self.num_of_labels = len(setlabel)

    def predict(self, X_test):
        predict = []
        num_doc_of_test = len(X_test)
        log_lamda_class = self.log_lamda_class
        log_pc = self.log_pc
        dim_fea_vector = self.dim_fea_vector
        labels = self.labels
        for i in range(num_doc_of_test):
            pretmp = np.zeros(self.num_of_labels)
            for j in range(self.num_of_labels):
                pretmp[j] += np.dot(log_lamda_class[j], X_test[i])
            pretmp += log_pc
            pos = np.array(pretmp).argmax()
            predict.append(labels[pos])
        return predict
