import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display 
import pandas as pd
from itables import show
import os 
import pandas as pd
import plotly.express as px
from itables import show

def display_png(image_path):
    img = mpimg.imread(image_path)  # Read the image file
    plt.imshow(img)  # Display the image
    plt.axis('off')  # Hide the axes
    plt.show()


def read_triples_from_file(file_path, draw = False, problem_type = "survival"):

    # Initialize variables
    triples = []
    current_triple = []

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Check if the line starts with CR, RR, or ER and add to the current triple
            if line.startswith("CR") or line.startswith("RR") or line.startswith("ER"):
                current_triple.append(line)
                
                # If we have collected a CR, RR, and ER in order, store it as a triple
                if len(current_triple) == 3:
                    # Check if it follows CR, RR, ER order
                    if current_triple[0].startswith("CR") and current_triple[1].startswith("RR") and current_triple[2].startswith("ER"):
                        triples.append(current_triple)
                    # Reset the current_triple to capture the next set
                    current_triple = []
            else:
                # Reset if line is blank or does not match expected rules
                current_triple = []

    if problem_type == "survival":
        image_name = "estymatory_"
    else:
        image_name = "histogram_"
    # Display the triples
    for i, triple in enumerate(triples, 1):
        display(f"Triple {i}:")
        for rule in triple:
            display(rule)

        if draw:
            # read png file
            triplet_number = triple[0][3:6].replace(":", " ").strip()
            display_png(file_path[:-9]+ f"{image_name}{triplet_number}.png")
        print()  # Blank line between triples




def read_triples_from_file_v2(file_path, draw = False, type_of_er = None):

    # Initialize variables
    triples = []  # List to store triples of CR, RR_i, ER_i sets
    current_set = []  # Temporary list to store a CR and its related RR_i and ER_i
    cr_rule = None  # To store the current CR rule
    rr_er_pairs = []  # List to store pairs of RR_i and ER_i for each CR

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Identify the start of a new CR rule
            if line.startswith("CR:"):
                # If there's a previous CR with RR/ER pairs, add it to triples
                if cr_rule and rr_er_pairs:
                    triples.append((cr_rule, rr_er_pairs))
                
                # Reset for the new CR rule
                cr_rule = line
                rr_er_pairs = []
            
            # Capture RR_i rules
            elif line.startswith("RR_"):
                rr_er_pairs.append((line, None))  # Placeholder None for the ER that will pair with this RR
            
            # Capture ER_i rules and types
            elif line.startswith("ER_"):
                # Extract the type of ER
                type_start = line.find("type")
                er_type = line[type_start:].split(":")[1].strip() if type_start != -1 else "None"
                
                # Update the last RR entry with the corresponding ER and its type
                if rr_er_pairs:
                    rr_er_pairs[-1] = (rr_er_pairs[-1][0], f"{line} (Type: {er_type})")
            
        # Append the last CR with its RR/ER pairs if any
        if cr_rule and rr_er_pairs:
            triples.append((cr_rule, rr_er_pairs))

    # Display the triples
    for i, (cr, pairs) in enumerate(triples, 1):
        first = True
        for rr, er in pairs:
            type_start = er.find("type")
            er_type = er[type_start:].split(":")[0].split(" ")[1]
            if (type_of_er is None) or (er_type == type_of_er):
                if first == True:
                    print(f"Triple {i}:")
                    print(cr)
                    first =False
                print(f"  {rr}")
                if er:
                    print(f"  {er}\n")
        # print()  # Blank line between triples


def plot_box(df, for_field, title, y_axis):
    fig = px.bar(df, x='dataset', color="type",
                y=for_field,
                barmode='group',
                #text_auto=True,
                #height=560,
                #width=3000,
                title=title,
                )
    fig.update_traces(textposition='outside')
    fig.update_layout(
                title={
                'x':0.5,
                'xanchor': 'center'
            })
    fig.update_layout(yaxis_title=y_axis)
    fig.show()

def get_table(df, wskaznik, return_df = True):
    df = df[["dataset", "type", wskaznik]]

    df = df.pivot(
        index="dataset", 
        columns= "type", 
        values=wskaznik
        )
    
    if return_df:
        return df
    else:
        show(df)

def prepare_data_for_plot(df):
    df_long = pd.melt(
        df,
        id_vars=["dataset"],
        value_vars=[
            "train_bacc_type_0", "test_bacc_type_0",
            "train_bacc_type_1", "test_bacc_type_1",
            "train_bacc_type_2", "test_bacc_type_2",
            "train_bacc_type_3", "test_bacc_type_3"
        ],
        var_name="type_tmp",
        value_name="value"
    )

    df_long["set"] = df_long["type_tmp"].str.split("_").str[0]  # train/test
    df_long["type"] = df_long["type_tmp"].str.split("_").str[-1]

    df_long = df_long.drop(columns=["type_tmp"])

    df_train = df_long[df_long["set"] == "train"].drop(columns=["set"])
    df_test = df_long[df_long["set"] == "test"].drop(columns=["set"])

    return df_train, df_test

def plot_box(df, for_field, title, y_axis, color="version"):
    fig = px.bar(df, x='dataset', color=color,
                y=for_field,
                barmode='group',
                #text_auto=True,
                #height=560,
                #width=3000,
                title=title,
                )
    fig.update_traces(textposition='outside')
    fig.update_layout(
                title={
                'x':0.5,
                'xanchor': 'center'
            })
    fig.update_layout(yaxis_title=y_axis)
    fig.show()

def get_table(df, wskaznik, return_df = False):
    df = df[["dataset", "version", wskaznik]]

    df = df.pivot(
        index="dataset", 
        columns= "version", 
        values=wskaznik
        )
    
    if return_df:
        return df
    else:
        show(df)