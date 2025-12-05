import csv

import numpy as np
import matplotlib.pyplot as plt


def load_timeline_csv(input_csv: str = "timeline_input.csv",
                      ) -> None:
    
    years = []
    events = []
    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
    
        for row in reader:
            years.append(int(row[0].strip()))
            events.append(row[1].strip())
    return years,events

if __name__== "__main__":
    #
    output_image = "topology_optimization_timeline.png"
    #
    years,events = load_timeline_csv()
    #
    fig, ax = plt.subplots(1,1,figsize=(13, 6))
    
    # draw central line
    ax.hlines(1, xmin=0, xmax=len(years)-1, color="black", linewidth=1)
    
    # scatter points
    ax.scatter(range(len(years)), 
               [1]*len(years), s=60)
    
    # alternate label heights for readability
    label_positions = []
    for i in range(len(years)):
        label_positions.append(1.1 if i % 2 == 0 else 0.9)
    
    # draw labels
    for year,i,event,ypos in zip(years, range(len(years)), events, label_positions):
        ax.plot([i,i], [1,ypos-np.sign(ypos-1)*0.01], 
                color="k", linestyle="--")
        ax.text(i, ypos, f"{year}\n{event}",
                ha="center", 
                va="bottom" if ypos > 1 else "top", 
                fontsize=11)
        
    #
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(int(year)) for year in years])
    ax.set_yticks([])
    ax.set_ylim(0.85,1.15)
    #
    ax.set_xlabel("Year")
    #ax.tight_layout()
    ax.axis("off")
    
    # Save and show
    plt.savefig(output_image, dpi=300,
                bbox_inches="tight")
    plt.show()