from typing import List
class EventGroup:
    def __init__(self,group_name: str, tag_label_pairs:dict):
        self.group_name =  group_name
        self.group = tag_label_pairs
        self.labels = list(tag_label_pairs.values())
        self.event_names = list(tag_label_pairs.keys())
        self.colors = ['#ee4035','#f37736','#fdf498','#7bc043','#0392cf','#f6db5f','#ffb554','#fe5e51','#9e3d64',
                        '#36abb5''#ffb3ba','#ffdfba','#ffffba','#baffc9','#bae1ff','#b7ded2','#f6a6b2','#f7c297','#ffecb8',
                        '#90d2d8','#ff71ce','#01cdfe','#05ffa1','#b967ff','#fffb96']
    def __call__(self):
        return self.group

    #def __repr__(self):
    #    string_repr = ["  - {} -> {}".format(event_name,label) for event_name,label in zip(self.event_names,self.labels)]
    ##    string_repr = '\n'.join(string_repr)
     #   return "{}:\n{}".format(self.group_name,string_repr)

class Taxonomy:
    def __init__(self, groups: List[EventGroup]):
        for g in groups:
            assert isinstance(g,EventGroup) , "{} is not of type EventGroup".format(g.group_name)
            setattr(self,g.group_name,g)
        self.groups = groups
        self.group_keys = [g.group_name for g in groups]
        assert len(self.group_keys) == len(groups)
        self.pool = {k: v for k, v in sorted({k: v for d in groups for k, v in d.group.items()}.items(), key = lambda item: item[1])}
        self.group_names = [g.group_name for g in groups]
        self.subclasses = [label for g in groups for label in g.labels]
        self.subnames = [label for g in groups for label in g.event_names]
        self.colors = [
                        '#ee4035',
                        '#f37736',
                        '#fdf498',
                        '#7bc043',
                        '#0392cf',

                        '#f6db5f',
                        '#ffb554',
                        '#fe5e51',
                        '#9e3d64',
                        '#36abb5'

                        '#ffb3ba',
                        '#ffdfba',
                        '#ffffba',
                        '#baffc9',
                        '#bae1ff',

                        '#ff71ce',
                       '#01cdfe',
                       '#05ffa1',
                       '#b967ff',
                       '#fffb96',

                       '#b7ded2',
                        '#f6a6b2',
                        '#f7c297',
                        '#ffecb8',
                        '#90d2d8',

                       ]
        self.colors = [
                        #'#FFEC1F',

                        '#FBAC23', #agn
                        '#F68128', #qso
                        '#D747CF', # EA
                        '#F25A2C', #yso
                        #####
                        '#ED3731', # cvnova
                        #
                        '#96034A', #blacar

                        '#BE4BD2',
                        '#8954C9',
                        '#554FCF',
                        '#2845E6',
                        "#3724EB",
                        '#1D2996',
                        '#1F78FF',
                        '#1FA9FF',
                        '#1FCEFF',
                        '#0CE9C0',
                        '#0CCA55',
                        '#2F8B04',
                        '#84FF1F',
                        '#FFC71F',

                        '#FFEC1F',
                        ]
    def __call__(self,group:str = None):
        if group is None:
            return self.pool
        elif group in self.group_keys:
            return self.__dict__[group]

    def __repr__(self):
        str_repr = [g.__repr__() for g in self.groups]
        str_repr = '\n'.join(str_repr)
        return str_repr

    def __len__(self):
        return len(self.pool.keys())

    def values_as_keys(self):
        return {value:key for key,value in self.pool.items()}



transient = EventGroup('transient',{"SNIa": 4,"SNII": 9,"SNIbc": 16,"SLSN": 17,"TDE": 18,"SNIIb": 19,"SNIIn": 20,"Microlensing": 21,})
stochastic = EventGroup('stochastic',{"AGN": 0,"QSO": 1,"YSO": 3,"CV/Nova": 5,"Blazar": 8,})
periodic = EventGroup( 'periodic', {"EA": 2,"RRLc": 6,"RSCVn": 7,"EB/EW": 10,"LPV": 11,"CEP": 12,"RRLab": 13,"Periodic-Other": 14,"DSCT": 15,})
ZTF_TAXONOMY = Taxonomy([transient, stochastic, periodic])


transient = EventGroup('transient',{"SNIa": 3,"SNIax": 1,"SNII": 5,"SNIbc": 4,"SLSN": 6,"TDE": 8,'KN':10, 'PISN':7, '91bg':2, 'ILOT':9})
stochastic = EventGroup('stochastic',{"AGN": 14,'Dwarf Novae':13,'M-dwarf Flare':11,'Microlens':12})
periodic = EventGroup('periodic',{"EB": 18,"RRL": 16, 'DeltaScuti':15,"Cepheid":17,"CART":0})
ELASTICC_TAXONOMY = Taxonomy([transient, stochastic, periodic])
