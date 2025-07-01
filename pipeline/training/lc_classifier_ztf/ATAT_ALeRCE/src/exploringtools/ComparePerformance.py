
class ComparePerformance:
    def __init__(self, 
                 path_1,
                 path_2,
                 path_to_dataset, 
                 model_class, 
                 model_type,  
                 custom_parse_key_str, 
                 seed = 0, 
                 device="cpu", 
                 batch_size=128, 
                 load_checkpoint=True ):
        self.taxonomy = None
        self.model_1 = ReportClassification(path_1, 
                                            path_to_dataset, 
                                            model_class, 
                                            model_type,  
                                            custom_parse_key_str, 
                                            seed = seed, 
                                            device=device, 
                                            batch_size=batch_size, 
                                            load_checkpoint=load_checkpoint )
        self.model_2 = ReportClassification(path_2, 
                                            path_to_dataset, 
                                            model_class, 
                                            model_type,  
                                            custom_parse_key_str, 
                                            seed = seed, 
                                            device=device, 
                                            batch_size=batch_size, 
                                            load_checkpoint=load_checkpoint )
    
    def compare(self, dataset_type:str= 'validation', eval_time = 2048):
        from sklearn.metrics import classification_report
        dataloader = self._init_dataloader(
            path_to_dataset=self.path_to_dataset, set_type=dataset_type, seed=self.seed
        )
        preds_1, target = self.model_1._predict(dataloader)
        preds_2,_ = self.model_2._predict(dataloader)
        cr_1  = classification_report(np.argmax(preds_1, axis = -1), target, output_dict=True)
        cr_2  = classification_report(np.argmax(preds_2, axis = -1), target, output_dict=True)
        f1_1 = []
        f1_2 = []
        fig,ax = plt.subplots(1,1,figsize = (8,5))
        for key,value in cr_1.items():
            if key in self.taxonomy().keys():
                #print(key,value)
                f1_1.append(value['f1-score'])
        df_1 = pd.DataFrame({'group':self.taxonomy().keys(), 'values':f1_1 })
        ordered_df_1 = df_1#.sort_values(by='values')
        my_range_1=range(1,len(df_1.index)+1)
        for key,value in cr_2.items():
            if key in self.taxonomy().keys():
                #print(key,value)
                f1_2.append(value['f1-score'])
        df_2 = pd.DataFrame({'group':self.taxonomy().keys(), 'values':f1_2})
        ordered_df_2 = df_2#.sort_values(by='values')
        my_range_2=range(1,len(df_2.index)+1)
        import matplotlib.pyplot as plt
        plt.plot(ordered_df_1['values'].values, my_range_1, "o", alpha = 0.8,color='blue')
        plt.plot(ordered_df_2['values'].values, my_range_2, "o", alpha = 0.8,color='red')
        plt.legend(['1','2'])
        # The horizontal plot is made using the hline function
        plt.hlines(y=my_range_1, xmin=0, xmax=ordered_df_1['values'], color='blue', alpha = 0.5)
        # The horizontal plot is made using the hline function
        plt.hlines(y=my_range_2, xmin=0, xmax=ordered_df_2['values'], color='red', alpha = 0.5)
        # Add titles and axis names
        plt.yticks(my_range_1, ordered_df_1['group'])
        plt.title(f"F1-Score LC Classifier Comparison for Eval Time {eval_time} ", loc='center')
        plt.xlabel('F1-Score')
        plt.ylabel('Class')
        plt.xlim(0,1)
        plt.grid('on', )
        # Show the plot
        plt.show()

