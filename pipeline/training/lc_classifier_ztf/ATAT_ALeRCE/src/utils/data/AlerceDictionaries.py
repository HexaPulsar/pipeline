from dataclasses import dataclass



@dataclass
class AlerceTaxonomy:
    transient: dict
    stochastic: dict
    periodic:dict
    def __post_init__(self):
        self.all_classes = {}
        self.all_classes.update(self.transient)
        self.all_classes.update(self.stochastic)
        self.all_classes.update(self.periodic)
        self.hierarchy_tree = {0:self.transient, 1:self.stochastic,2:self.periodic}
        
       
        self.colors =  ["#43aa8b",
                        "#277da1",
                        "#ca5cdd",
                        "#277da1",
                        "#f9c74f",
                        "#90be6d",
                        "#f8961e",
                        "#f94144",
                        "#f9844a",
                        "#ca5cdd",
                        "#f3722c",
                        "#277da1",
                        "#43aa8b",
                        "#577590",
                        "#4d908e",
                        "#f9c74f",
                        "#90be6d",
                        "#f94144",
                        "#f3722c",
                        "#f8961e",
                        "#277da1",
                        "#ca5cdd",
                    ]
    def list_leaf_class_names(self):
        pass
    
    def list_node_class_names(self):
        pass
    def get_parent(class_:int):
        pass
    def get_children(class_:int):
        pass
    
    def __len__(self):
        return sum([len(self.transient.values()),len(self.stochastic.values()),len(self.periodic.values())])
    
    def __getitem__(self,key):
        """returns tuple (node_class, leaf_class, class_name)

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (self.values_as_keys(self.hierarchy_tree),key,self.values_as_keys(self.all_classes)[key])
    @staticmethod
    def values_as_keys(dict_):
        """inverts an arbitrary dictionary {keys:values} --> {values:keys}

        Returns:
            dict: value:key dictionary
        """
        print(dict_)
        return {value:key for key,value in dict_.items()}
    
    def node_leaf_tuple(self):
        for key, value in self.hierarchy_tree:
            print(key,value)
    
ALERCE_TAXONOMY = AlerceTaxonomy({"SNIa": 4,"SNII": 9,"SNIbc": 16,"SLSN": 17,"TDE": 18,"SNIIb": 19,"SNIIn": 20,"Microlensing": 21,},
                                 {"AGN": 0,"QSO": 1,"YSO": 3,"CV/Nova": 5,"Blazar": 8,},
                                 {"EA": 2,"RRLc": 6,"RSCVn": 7,"EB/EW": 10,"LPV": 11,"CEP": 12,"RRLab": 13,"Periodic-Other": 14,"DSCT": 15,})
