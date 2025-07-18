# toptoptlab 

This project is a collection of topology optimization techniques many of which
are already published in Matlab. This project is there to make these techniques
available to people without Matlab licenses but also to offer a basis on which
new methods can be implemented quickly. Please be aware that Matlab is 
optimized for tasks like topology optimization while Python is a general 
purpose language, so you will likely see slower performance here than in 
equivalent Matlab scripts.

# how to use 

Topoptlab can be used either as a black box function that returns an optimal 
design to you once provided with a set of boundary conditions and parameter or 
as a general purpose topology optimization and finite element library.


# Monolithic codes

In the monolithic_code directory, you can find codes that are self contained 
and do not use the topoptlab module written here (at the moment not true). 
These codes are purely there to either test new frameworks (e. g. JAX or PETSC)
or for teaching/demonstration purposes. If you are completely new to topology 
optimization, this is where you start and I suggest to start with the topopt.py.

# installation
Basic installation and run tests by executing
```
pip install .[tests]
```
in top directory. Editable installation (recommended if you want to edit 
something in the code) 
```
pip install -e .[tests]
```

# run tests
Run fast tests (finish in under one minute)
```
pytest
```
Run slow tests (take a few minutes)
```
pytest -m slow
```

# Roadmap

See [ROADMAP.md](./ROADMAP.md) for a list of upcoming features.

# make documentation

The documentation can be build via Sphinx 

```
cd docs/
sphinx-apidoc -o source/ ../topoptlab/ ../topoptlab/legacy --force --no-toc --separate
make html
```
and displayed in your browser by drag and drop or if you are on Linux
```
xdg-open build/html/index.html
```

# getting report bugs, and suggest enhancements

If you found a bug or you want to suggest a new feature/enhancement, submit it on the [issue tracker](https://github.com/stefanhiemer/topoptlab/).

# wow to contribute

If you want to contribute, fork the repository and open a pull request. However before doing that
I suggest to contact the maintainers via an enhancement suggestion in the issue tracker (see above).