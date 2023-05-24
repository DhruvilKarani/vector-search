import os

if os.path.exists('../vectors'):
    os.system('rm -rf ../vectors')
os.system('mkdir ../vectors')

# download ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz in ../vectors using wget
os.system('wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -P ../vectors')
os.system('tar -xzf ../vectors/sift.tar.gz -C ../vectors')