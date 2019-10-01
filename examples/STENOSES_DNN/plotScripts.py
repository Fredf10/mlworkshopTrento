import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
def plotScatterGeneric(x ,y ,xN, yN, name, savePlots=True):
    
    xdom = np.linspace(x.min(),x.max(),100)
    avg = np.average(y)
    
    rho = pearsonr(x, y)

    fit = np.polyfit(x, y, deg=1)
    
    #plt.axhline(y=avg,color='k')
    std = np.std(y)
    plt.figure()
    tit = 'Slope: %.2f; Intercept: %.2f; Corr: %.2f; Nr.: %i\n' % (fit[0],fit[1],rho[0], len(x))
    #tit = 'Mean diff:  %.3f; Std. Dev.: %.3f; Nr.: %i' % (avg,std,len(x))
    plt.title(tit)
    plt.scatter(x,y,facecolors='none', edgecolors='k')
    plt.xlabel(xN)
    plt.ylabel(yN)
    plt.plot(xdom,xdom,'k--')
    #plt.ylim(0,5)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.tight_layout()
    if savePlots:
        plt.savefig(name)
        plt.close()
#===================================================
    
def plotBAGeneric(x, y, xN, yN, name, savePlots=True):
    
    diff = (x-y)
    plt.figure()
    plt.scatter(x,diff,facecolors='none', edgecolors='k')#,facecolors='none', edgecolors='k')
    plt.xlabel(xN)
    plt.ylabel(yN)
    avg = np.average(x-y)
    plt.axhline(y=avg,color='k')
    std = np.std(x-y)
    #tit = 'Mean diff:  %.3f; Std. Dev.: %.3f; Nr.: %i' % (avg,std,len(x))
    plt.axhline(avg+2.*std,color='k',ls='--')
    plt.axhline(avg-2.*std,color='k',ls='--')
    #plt.title(tit)
    tit = 'Mean diff:  %.3f; Std. Dev.: %.3f; Nr.: %i' % (avg,std,len(x))
    plt.title(tit)
    plt.tight_layout()
    if savePlots:
        plt.savefig(name,filetype='png')
        plt.close()
#===================================================

def plotGeom(x, r, figFolder, stenoCase, trainOrTest):
    createDirIfEmpty(figFolder)
    figFolder = os.path.join(figFolder, stenoCase)
    createDirIfEmpty(figFolder)
    figFolder = os.path.join(figFolder, trainOrTest)
    createDirIfEmpty(figFolder)
    
    figName = os.path.join(figFolder, "ori_geom.png")
    plt.figure()
    plt.plot(x, r)
    plt.ylim((0, 1.1*max(r)))
    plt.xlabel("x [cm]")
    plt.ylabel("x [cm]")
    plt.title(stenoCase + "_" + trainOrTest)
    plt.savefig(figName)
#===================================================

if __name__ == '__main__':
    
    
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    
    plotScatterGeneric(a ,b ,"v", "v", "test.png")
    
    
