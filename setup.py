from setuptools import find_packages, setup

PACKAGES = find_packages()

setup(
    name='david',
    packages=PACKAGES,
    version='0.0.1',
    url='https://github.com/amoux/david',
    author='Carlos A. Segura Diaz De Leon',
    author_email='carlosdeveloper2@gmail.com',
    license='MIT',
    zip_safe=False,
    scripts=[
        'bin/download-spacy-models',
        'bin/download-spacy-elmo-x2',
    ]
)
