import numpy as np
import pandas as pd
from torch.utils.data import Sampler, Subset
from collections import defaultdict

# Define a custom batch sampler to ensure each batch contains data from a single dataset_idx group
class GroupBatchSampler(Sampler):
    def __init__(self, subset):
        # Print a message to show that the function is running
        print()        
        print("Inside __init__ function")
        print("==========================================================================================")

        self.subset = subset
        # Print out the subset itself to show what it looks like
        print(f"Inside Class self.subset: {self.subset}")
        print()

        self.dataset = subset.dataset
        self.indices = subset.indices
        # Print out the first 5 elements of the dataset and indices
        print(f"Inside Class self.dataset: {self.dataset}")
        print(f"Inside Class self.indices: {self.indices}")
        print()

        # Print out the first 5 elements of the dataset in a readable format
        print("First 5 elements of the self.dataset in a readable format: ")
        for idx in self.indices[:5]:
            print(f"idx: {idx}, self.dataset[idx]: {self.dataset[idx]}")
        print()

        self.group_indices = self._get_group_indices()
        print(f"self.group_indices after initialization: {self.group_indices}")

        # Print a message to show that the function has finished running
        print("Finished __init__ function")
        print("==========================================================================================")
        print()
    
    def _get_group_indices(self):
        # Print a message to show that the function is running
        print()
        print("Inside _get_group_indices function")
        print("==========================================================================================")

        group_indices = defaultdict(list)
        # Print out what the group_indices look like before the loop in a readable format
        print(f"group_indices before loop: {group_indices}")
        print()

        for subset_idx, dataset_idx in enumerate(self.indices):
            # Print out the first 5 subset_idx and dataset_idx values, what they grab from self.indices and from the dataset.
            # Create a variable that starts at 0 and increments by 1 each time the loop runs, and terminate the printing after 5 loops.
            if subset_idx < 5:
                print(f"subset_idx: {subset_idx}, dataset_idx: {dataset_idx}")
                print(f"Accessing self.dataset[dataset_idx]: {self.dataset[dataset_idx]}")
                print()

            group = self.dataset.dataset_idx[dataset_idx].item()
            if subset_idx <= 0:
                print("NOTE**************************************")
                print("group is accessed by self.dataset.dataset_idx[dataset_idx].item()")

            # For the first 5 subset_idx values, print out the group value
            if subset_idx < 5:
                # Print out the entire place that the group value is stored
                print(f"self.dataset.dataset_idx[dataset_idx]: {self.dataset.dataset_idx[dataset_idx]}")
                print(f"group: {group}")
                print()

            group_indices[group].append(subset_idx)
        # Print out what the group_indices look like after the loop in a readable format
        print(f"Raw group_indices after loop: {group_indices}")
        print()

        # Now print out the first 5 rows of the subset ordered by group_indices in a dataframe to show what it looks like
        print("The first 5 rows of the subset with group_indices in a DataFrame: ")
        df = pd.DataFrame({
            'self.dataset[idx]': [self.dataset[idx] for idx in range(5)],
            'self.dataset.dataset_idx[idx].item()': [self.dataset.dataset_idx[idx].item() for idx in range(5)]
        })
        print(df)
        print()


        # Print a message to show that the function has finished running
        print("Finished _get_group_indices function")
        print("==========================================================================================")
        print()

        return list(group_indices.values())

    def __iter__(self):
        # Print a message to show that the function is running
        print()
        print("Inside __iter__ function")
        print("==========================================================================================")

        np.random.shuffle(self.group_indices)
        print(f"Group indices after shuffling: {self.group_indices}")
        for group in self.group_indices:
            yield group
        
        # Print a message to show that the function has finished running
        print("Finished __iter__ function")
        print("==========================================================================================")
        print()

    def __len__(self):
        # Print a message to show that the function is running
        print()
        print("Inside __len__ function")
        print("==========================================================================================")
        return len(self.group_indices)

# Example usage
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, Dataset

    # Dummy dataset class
    class DummyDataset(Dataset):
        def __init__(self):
            print("Inside DummyDataset __init__ function")
            print("==========================================================================================")
            print("==========================================================================================")
            




            self.data = torch.randint(0, 1000, (100,))  # Random numbers between 0 and 1000
            print("self.data = torch.randint(0, 1000, (100,))  # Random numbers between 0 and 1000")
            print()

            print("Debug Notes: ---------------------------------------------------------------------------------")
            # Print the first 5 elements of the dataset
            print("for idx in range(5): ")            
            for idx in range(5):
                print(f"idx: {idx}, self.data[idx]: {self.data[idx]}")
            print("---------------------------------------------------------------------------------------------")
            print()



            


            # Generate random numbers between 0 and 5 to represent the dataset index
            self.dataset_idx = torch.randint(0, 5, (100,))
            print("self.dataset_idx = torch.randint(0, 5, (100,)) # 100 random numbers between 0 and 5 generate to represent the dataset index")
            
            print("Debug Notes: ---------------------------------------------------------------------------------")
            # Print the first 5 elements of the dataset_idx
            print("for idx in range(5): ")
            for idx in range(5):
                print(f"idx: {idx}, self.dataset_idx[idx]: {self.dataset_idx[idx]}")
            print("---------------------------------------------------------------------------------------------")
            print()






            # Create a DataFrame to visualize the data and indices.
            # NOTE: When you create an instance of DummyDataset, the __init__ method is called.
            # NOTE: Inside this method, self.data and self.dataset_idx are created as instance variables.
            # NOTE: These variables are accessible throughout the instance of the class.
            df = pd.DataFrame({
                'Data': self.data.numpy(),
                'Dataset Index': self.dataset_idx.numpy()
            })
            print("df = pd.DataFrame({")
            print("    'Data': self.data.numpy(),")
            print("    'Dataset Index': self.dataset_idx.numpy()")
            print("})")

            print("NOTES: _____________________________________________________________________________________________")
            print("     • When you create an instance of DummyDataset, the __init__ method is called.")
            print("     • Inside this method, self.data and self.dataset_idx are created as instance variables.")
            print("     • These variables are accessible throughout the instance of the class.")
            print("     • After the instance is created, you can access these variables using the instance name.")
            print("     • So when an instance of DummyDataset is created called 'dataset', ")
            print("       you can access the data and dataset_idx using 'dataset.data' and 'dataset.dataset_idx'.")
            print("___________________________________________________________________________________________________")
            print()







            print("Debug Notes: ---------------------------------------------------------------------------------")
            print("Dataset Table Created:")
            print(df)
            print("---------------------------------------------------------------------------------------------")
            print()

            print("Finished DummyDataset __init__ function")
            print("==========================================================================================")
            print("==========================================================================================")
            print()







        def __len__(self):
            print("Inside DummyDataset __len__ function")
            print("==========================================================================================")
            print("==========================================================================================")

            print("return len(self.data)")
            print()


            print("Debug Notes: ---------------------------------------------------------------------------------")
            print(f"Length of the dataset: {len(self.data)}")
            print("All this function does is return the length of the dataset.")
            print("---------------------------------------------------------------------------------------------")


            print("Finished DummyDataset __len__ function")
            print("==========================================================================================")
            print("==========================================================================================")
            print()
            return len(self.data)
        






        def __getitem__(self, idx):
            # NOTE: This __getitem__ method returns only the self.data[idx] and not the dataset_idx
            print("Inside DummyDataset __getitem__ function")
            print("==========================================================================================")
            print("==========================================================================================")
            print()

            print("return self.data[idx]")
            print()
            
            
            print("NOTES: _____________________________________________________________________________________________")
            print("     • The __getitem__ method is used to access 'data' at a specific index.")
            print("     • This function is called when you use the index operator on an instance of the class.")
            print("     • So when you use 'dataset[idx]', this function is called.")
            print("          ○ That means that when you use 'dataset[idx]', you only get access to 'data' and not 'dataset_idx'.")
            print("          ○ If you want to access 'dataset_idx', you need to access it directly using 'dataset.dataset_idx'.")
            print("          ○ You could also access 'data' by using 'dataset.data'.")
            print("     • To clarify what 'dataset', 'data', and 'dataset_idx' are:")
            print("          ○ 'dataset' is an instance of the DummyDataset class.")
            print("          ○ 'data' is an instance variable of the DummyDataset class that contains random numbers between 0 and 1000.")
            print("          ○ 'dataset_idx' is an instance variable of the DummyDataset class that contains random numbers between 0 and 5.")
            print("_____________________________________________________________________________________________________")
            print()

            



            print("Finished DummyDataset __getitem__ function")
            print("==========================================================================================")
            print("==========================================================================================")
            print()
            return self.data[idx]







    # Create a dataset and a subset
    dataset = DummyDataset()
    print("dataset = DummyDataset()")
    print()

    print("Debug Notes: ---------------------------------------------------------------------------------")
    print("'dataset' is an instance of the DummyDataset class.")
    print("This means that 'dataset' has access to all the methods and variables in the DummyDataset class.")
    print("You can access the data and dataset_idx using 'dataset.data' and 'dataset.dataset_idx'.")
    # Call the 'data' and 'dataset_idx' variables from the 'dataset' instance to show that they are accessible
    print("dataset.data: ", dataset.data)
    print("dataset.dataset_idx: ", dataset.dataset_idx)
    print()

    print("If you use 'dataset[idx]', the __getitem__ method is called, and you only get access to 'data' and not 'dataset_idx'.")
    # Call the __getitem__ method using the 'dataset' instance to show that it only returns the 'data' value
    print("dataset[0]: ", dataset[0])
    print("---------------------------------------------------------------------------------------------")







    indices = list(range(50))  # Subset of the first 50 elements
    print("indices = list(range(50))  # This creates a list from 0 to 49 which is used to create a subset of the first 50 elements.")
    print()










    subset = Subset(dataset, indices)
    print("subset = Subset(dataset, indices)")
    print()

    print("Debug Notes: ---------------------------------------------------------------------------------")
    print(" When you create a 'Subset' object using 'Subset(dataset, indices)', it creates a new object that references the original dataset and the specified indices.")











    # Print out the entire first 5 elements of the subset, and note that the 'subset' does not contain the dataset_idx, and
    # explain how the 'Subset' class works.
    print("The first 5 elements of 'subset' will be printed out below.")
    print("Note that the 'subset' does not contain the dataset_idx.")
    print("The 'Subset' class is a class that represents a subset of a dataset at specified indices and does not do anything else.")
    print("It is used to create a subset of a dataset without changing the dataset itself.")
    for idx in range(5):
        print(f"idx: {idx}, subset[idx]: {subset[idx]}")
    print()

    # Now print the first 5 rows of 'subset' as a DataFrame to show what it looks like
    print("The first 5 rows of 'subset' as a DataFrame:")
    df = pd.DataFrame({
        'Data': [subset[idx] for idx in range(5)]
    })
    print(df)
    # Print a message to note that the Data column contains tensor values, and the number inside 'tensor()' is the data value that
    # corresponds to the index in the 'subset', which is the same as the index in the 'dataset'.
    print("Note that the Data column contains tensor values.")
    print("The number inside 'tensor()' is the data value that corresponds to the index in the 'subset', which is the same as the index in the 'dataset'.")
    print("So what subset does is it simply grabs the data values from the dataset at the specified indices.")
    print("The dataset_idx is not included in the subset because the 'Subset' class can only grab one thing from the dataset, and it grabs the data.")
    print()

    # Create the custom batch sampler
    batch_sampler = GroupBatchSampler(subset)

    # Create a DataLoader with the custom batch sampler
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(f"Batch: {batch}")