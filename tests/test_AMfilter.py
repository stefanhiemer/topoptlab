from numpy import array,eye,triu,loadtxt,savetxt
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.filters import AMfilter

@pytest.mark.parametrize('bplate',
                         [('N'),('E'),('S'),('W')])

def test_eye_density(bplate):
    #
    x = (eye(5)[:,:4]+1)/2
    assert_almost_equal(AMfilter(x,bplate),
                        loadtxt("./test_files/AMfilter_eye-density-"+str(bplate)+".csv",
                                delimiter=",")) 
    return

def test_eyetriu_density(bplate):
    #
    x = triu((eye(5)[:,:4]+1)/2)
    assert_almost_equal(AMfilter(x,bplate),
                        loadtxt("./test_files/AMfilter_eyetriu-density-"+str(bplate)+".csv",
                                delimiter=",")) 
    return


def test_eye_sensitivity(bplate):
    #
    x = (eye(5)[:,:4]+1)/2
    assert_almost_equal(AMfilter(x,bplate,x[:,:,None]),
                        loadtxt("./test_files/AMfilter_eye-sensitivity-"+str(bplate)+".csv",
                                delimiter=",")[:,:,None]) 
    return

def test_eyetriu_sensitivity(bplate):
    #
    x = triu((eye(5)[:,:4]+1)/2)
    assert_almost_equal(AMfilter(x,bplate,x[:,:,None]),
                        loadtxt("./test_files/AMfilter_eyetriu-sensitivity-"+str(bplate)+".csv",
                                delimiter=",")[:,:,None]) 
    return

def test_random_density(bplate):
    
    x = array([[2.672288588974325e-01, 7.225472722386593e-01, 6.992762490789783e-01, 6.524823915201646e-01], 
               [1.728954593931380e-01, 1.795476379359208e-01, 9.887395086857194e-01, 4.865503104248466e-01],
               [7.725985923254057e-01, 5.597233640757759e-02, 2.194182791613052e-01, 2.227367832399215e-01],
               [4.726430926198782e-01, 5.104272798058791e-01, 1.378729341440889e-01, 3.429295050273223e-01],
               [7.307387670425288e-01, 5.135599465352547e-01, 6.158474764756328e-02, 5.290986163088152e-01]
               ])
    
    assert_almost_equal(AMfilter(x,bplate), 
                        loadtxt("./test_files/AMfilter_random-density-"+bplate+".csv",
                                delimiter=","))
    
    return

def test_random_sensitivity(bplate):
    
    x = array([[2.672288588974325e-01, 7.225472722386593e-01, 6.992762490789783e-01, 6.524823915201646e-01], 
               [1.728954593931380e-01, 1.795476379359208e-01, 9.887395086857194e-01, 4.865503104248466e-01],
               [7.725985923254057e-01, 5.597233640757759e-02, 2.194182791613052e-01, 2.227367832399215e-01],
               [4.726430926198782e-01, 5.104272798058791e-01, 1.378729341440889e-01, 3.429295050273223e-01],
               [7.307387670425288e-01, 5.135599465352547e-01, 6.158474764756328e-02, 5.290986163088152e-01]
               ])
    
    assert_almost_equal(AMfilter(x,bplate,x[:,:,None])[:,:,0],
                        loadtxt("./test_files/AMfilter_random-sensitivity-"+bplate+".csv",
                                delimiter=","))
    return


    
