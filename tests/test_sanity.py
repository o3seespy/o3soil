import o3seespy as o3


def test_init_o3():
    osi = o3.OpenSeesInstance(ndm=2)
    o3.wipe(osi)


