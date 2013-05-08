# Move all the USpop labels into a single directory, keeping only the files we need

import os, re

os.chdir('/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/reformat_uspop/')

root_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/USpoplabs/uspopLabels/'
new_dir = '/Users/mattmcvicar/Desktop/Work/New_chroma_features/Package/USpoplabs_flat/'

server_root = '/u/drspeech/data/music/mp3s/'
server_names = open('./server_names.txt','w+')
for path, subdirs, files in os.walk(root_dir):
    for name in files:
        if os.path.splitext(name)[1] == '.lab':
          # Need artist name
          artist = re.split('/',path)[-2]    
          
          # Trim number from name
          name_nonum = name[3:]
          os.system('cp ' + os.path.join(path, name) + ' ' + new_dir + artist + '_-_' +  name_nonum)
          
          # Also get the name on the server
          server_name = server_root + '/'.join(re.split('/',path)[-2:]) + '/' + name[:-len('lab')] + 'mp3'
          server_names.write(server_name + '\n')
          
server_names.close()          