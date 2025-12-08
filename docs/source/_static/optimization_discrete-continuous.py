import numpy as np
import matplotlib.pyplot as plt

def continuous_vs_discrete():
    # create data
    x_cont = np.linspace(-2*np.pi, 2*np.pi, 401)
    y_cont = -np.sinc(x_cont)

    # discrete
    x_disc = np.linspace(-2*np.pi, 2*np.pi, 31)
    y_disc = -np.sinc(x_disc)

    # 
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x_cont, y_cont, label="continuous")
    ax.scatter(x_disc, y_disc, label="discrete")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("sinc(x)", fontsize=16)
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    plt.savefig("discrete-vs-continuous.png", format="png", 
                bbox_inches="tight")

    plt.show()
    return

def convex_set():
    
    # create data
    theta = np.linspace(0, 2*np.pi, 400)
    #
    a, b = 2, 1   # ellipse radii
    x_convex = a * np.cos(theta)
    y_convex = b * np.sin(theta)
    # 
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(x_convex, y_convex, 
                    alpha=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("connected-space-1.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    #
    r = 1 + 0.3 * np.sin(5 * theta)  # radius oscillates to create non-convexity
    x_nonconvex = r * np.cos(theta)
    y_nonconvex = r * np.sin(theta)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(x_nonconvex, y_nonconvex, 
                    alpha=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("connected-space-2.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    #
    theta = np.linspace(0, 2*np.pi, 400)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(-1 + 0.8 * np.cos(theta), 0 + 0.8 * np.sin(theta),  
                    alpha=0.5, color="b")
    ax.fill_between(1 + 0.8 * np.cos(theta), 0 + 0.8 * np.sin(theta),  
                    alpha=0.5, color="b")
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("connected-space-3.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    return

def convex_function():
    
    # create data
    x = np.linspace(-2, 2, 401)
    # 
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, x**2, linestyle="--",color="k", linewidth=4)
    ax.fill_between(x, x**2 , y2=4, alpha=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("convex-function-1.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    #
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, np.zeros(x.shape), linestyle="--",color="k", linewidth=4)
    ax.fill_between(x, 4 -x**2 , y2=0, alpha=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("convex-function-2.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    return

def jensens_ineq():
    #
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.linspace(-2, 2, 401)
    ax.plot(x, x**2, linestyle="-",color="k", linewidth=4)
    #
    x = np.linspace(0,1,101)
    ax.plot(x, x, linestyle="--",color="b", linewidth=2)
    #
    x = np.linspace(-1,1.5,251)
    ax.plot(x, x/2 + 1.5, linestyle="--",color="b", linewidth=2)
    #
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("jensen-1.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    #
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.linspace(-2, 2, 401)
    mask = x<-1
    ax.plot(x[mask], 2*np.abs(x[mask])-1, linestyle="-",color="k", linewidth=4)
    mask = x>1
    ax.plot(x[mask], 2*np.abs(x[mask])-1, linestyle="-",color="k", linewidth=4)
    mask = (x<-1) | (x>1)
    ax.plot(x[~mask], np.abs(x[~mask]), linestyle="-",color="k", linewidth=4)
    #
    x = np.linspace(-1,1,101)
    ax.plot(x, np.ones(x.shape), linestyle="--",color="b", linewidth=2)
    #
    x = np.linspace(-1,2,251)
    ax.plot(x, 2/3*x + 5/3, linestyle="--",color="b", linewidth=2)
    #
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("jensen-2.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    return

def first_order_cond():
    #
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.linspace(-2, 2, 401)
    ax.plot(x, np.abs(x**3), linestyle="-",color="k", linewidth=4)
    #
    x = np.linspace(-1,1,101)
    ax.plot(x, np.zeros(x.shape), linestyle="--",color="b", linewidth=2)
    #
    x = np.linspace(0,2.,201)
    ax.plot(x, 3*x - 2, linestyle="--",color="b", linewidth=2)
    #
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("firstorder-1.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    #
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.linspace(-2, 2, 401)
    mask = x<-1
    ax.plot(x[mask], (x[mask]+1)**2, linestyle="-",color="k", linewidth=4)
    mask = x>1
    ax.plot(x[mask], (x[mask]-1)**2, linestyle="-",color="k", linewidth=4)
    mask = (x<-1) | (x>1)
    ax.plot(x[~mask], np.zeros(x[~mask].shape), linestyle="-",color="k", linewidth=4)
    #
    x = np.linspace(-2,2,101)
    ax.plot(x, np.zeros(x.shape), linestyle="--",color="b", linewidth=2)
    #
    x = np.linspace(0.5,2,201)
    ax.plot(x, x - 1.25 , linestyle="--",color="b", linewidth=2)
    #
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("firstorder-2.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    return

def second_order_cond():
    #
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.linspace(-2, 2, 401)
    ax.plot(x, x, linestyle="-",color="k", linewidth=4)
    #
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("secondorder-1.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    #
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.linspace(-2, 2, 401)
    ax.plot(x, x**2, linestyle="-",color="k", linewidth=4)
    #
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    ax.axis("off")
    plt.savefig("secondorder-2.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    return

def compactness():
    #
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.linspace(-10, 10, 401)
    ax.plot(x, np.exp(x), linestyle="-",color="k", linewidth=4)
    #
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    plt.savefig("compactness-1.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    #
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.linspace(-0, 5, 401)
    ax.plot(x, np.log(x), linestyle="-",color="k", linewidth=4)
    #
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    plt.savefig("compactness-2.png", format="png", 
                bbox_inches="tight")
    plt.show()
    plt.close()
    
    return


if __name__ == "__main__":
    
    #continuous_vs_discrete()
    #convex_set()
    #convex_function()
    #jensens_ineq()
    #first_order_cond()
    #second_order_cond()
    compactness()