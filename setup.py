from setuptools import setup, find_packages

about = {}
with open("o3soil/__about__.py") as fp:
    exec(fp.read(), about)

with open('README.rst') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

setup(name='o3soil',
      version=about['__version__'],
      description='A toolkit of extensions for using o3seespy for geotechnical simulations',
      long_description=readme,  # + '\n\n' + history,
      url='',
      author=about['__author__'],
      author_email='maxim.millen@canterbury.ac.nz',
      license=about['__license__'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Programming Language :: Python :: 3',
      ],
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      install_requires=[
        "numpy",
        "o3seespy"
      ],
      # List additional groups of dependencies here (e.g. development
      # dependencies). You can install these using the following syntax,
      # for example:
      # $ pip install -e .[dev,test]
      extras_require={
          'test': ['pytest'],
      },
      python_requires='>=3',
      package_data={},
      zip_safe=False)


# From python packaging guides
# versioning is a 3-part MAJOR.MINOR.MAINTENANCE numbering scheme,
# where the project author increments:

# MAJOR version when they make incompatible API changes,
# MINOR version when they add functionality in a backwards-compatible manner, and
# MAINTENANCE version when they make backwards-compatible bug fixes.