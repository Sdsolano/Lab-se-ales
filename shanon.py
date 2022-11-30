def ES(st):
    vt=st*0
    l=len(vt)
    for i in range(0,l):
        if st[i]==0:
            vt[i]=0
        else: 
            vt[i]=-(st[i]**2)*np.log(st[i]**2)
    #xbarra=np.mean(vt)
    return vt 
