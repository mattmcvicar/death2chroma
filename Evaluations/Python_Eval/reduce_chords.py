def reduce_chords(chords,alphabet):
  # Reduce a list of chords to a simpler alphabet

  reduced_chords = chords
  if alphabet == 'minmaj':
    reduce_to_minmaj(chords)
  if alphabet == 'triads':
    pass
  if alphabet == 'quads':
    pass

  return reduced_chords
  
# Mid-level functions to reduce the chords  
def reduce_to_minmaj(chords):
    
  reduced_chords = []

  for chord in (chords):
    
    quality = chord2quality(chord)
        
    if quality == 0:
      # major
      c_type='maj'
    if quality == 1: 
      # major
      c_type='maj'    
    if quality == 2:
      # major
      c_type='maj'    
    if quality == 3:
      # major
      c_type = 'maj'
    if quality == 4:  
       # suspended
       c_type = 'maj'
       
    # get rootnote and append type
    rootnote = getchordinfo(chord);
    
    reduced_chords.append(rootnote + ':' + c_type)

# Low level functions for extracting notes etc
def chord2quality(chordsymbol,verbose=0):

  quality = 0;
  errormessage = '';

  [rootnote,shorthand,degreelist,bassdegree,success,error] = getchordinfo(chordsymbol);

  if success:  
    if len(shorthand) > 0:  
        [quality, success, error] = short2quality(shorthand);      
    else:
        [quality, success, error] = degrees2quality(degreelist);
  else: 
    errormessage = 'Error in Chord2quality: Chord "' + chordsymbol
    if verbose == 1:
        print errormessage
        
  return  quality, success, errormessage       
   
def getchordinfo(chordsymbol, verbose=0):
    
  errormessage = '';
  rootnote = '';
  shorthand =  '';
  #degrees = '';
  #bassdegreel = '';


  success = 0;

  # parse the chord symbol into its constituent parts
  [rootnote,shorthand, degreelist,bassdegree, success, errormessage] = parsechord(chordsymbol);

  print rootnote
  print shorthand
  print degreelist
  print bassdegree
  sd;lkgfdklj
  if success:
    if rootnote is not 'N':
        
        # check validity of rootnote
        [temp, temp2, success, errormessage] = parsenote(rootnote)

        # check validity of shorthand list
        if success and len(shorthand) > 0:
           [temp, success, errormessage] = shorthand2degrees(shorthand);
        
        # check validity of degreelist
        if success and len(degreelist) > 0:
            [temp, success, errormessage] = parsedegreelist(degreelist);
    
        # check validity of bass degree
        if success and len(bassdegree) > 0:
           [temp,temp2,temp3, success, errormessage] = parsedegree(bassdegree); 

  if not success and verbose:
    print errormessage
    
  return rootnote, shorthand,degreelist,bassdegree, success, errormessage

def parsechord(chord, verbose=0):

  ilength = len(chord);

  # initialise variables
  errormessage = '';
  rootnote = '';
  shorthand = '';
  degrees = '';
  bass = '';

  success = True;
  index = 1;

  # check for 'no chord' symbol
  if chord[index-1] == 'N':
    rootnote = chord[index-1]
    index = index + 1;
    # check to see there are no further characters
    if (index <= ilength):
        print 'Error in parsechord: Extra characters after "no chord" symbol' 
        success = 0;
  else:
  # parse thechord symbol

    # the first part of the symbol before a switch character should be the root note
    while (index <= ilength):
    
      if any([chord[index-1]==':', chord[index-1]=='/', chord[index-1]=='(', chord[index-1] ==  ')']):
            break

      rootnote = rootnote + chord[index-1]
      index = index + 1
    
      if (index > ilength) or chord[index-1] == '/':
        # if chord is a rootnote on its own or with just a bass note 
        # then it is a major chord therefore set shorthand to 'maj'
        shorthand = 'maj';
    
    # initialise booleans to record which switch characters we have found
    colon = False
    openbracket = False
    closebracket = False
    slash = False
    
    # parse the rest of the chord symbol
    while(index <= ilength):
      print index
      print ilength
        
      # reset temporary index 
      tempindex = 1
        
      if chord[index-1] == ':':
        
        # if we find a colon after any switch characters have 
        # already occured then the symbol is incorrect 
        if any([colon, openbracket, closebracket, slash]):
          print 'Error in parsechord: Incorrect character sequence in chord "' + chord 
          success = 0
          index = ilength + 1
        else:         
          # found the first instance of a colon character
          colon = True
          index = index + 1
                   
          if (index > ilength):
            print 'Error in parsechord: Found ":" at end of chord string "' + chord 
            success = 0
                   
          # colon should be followed by a shorthand string or
          # a degree list contained in brackets
          while (index <= ilength):
                       
            if any([chord[index-1]==':', chord[index-1]=='/', chord[index-1]=='(', chord[index-1]==')']):
              break

            # copy character into shorthand
            shorthand = shorthand + chord[index-1]
            index = index + 1
            tempindex = tempindex + 1
                     
      elif chord[index-1] == '(':
               
        # if we have had a colon but no other switch charaters then
        # an open bracket signifies the start of the degree list
        if all([colon, not slash, not closebracket, not openbracket]): 
          openbracket = True
          index = index + 1
                   
          while (index <= ilength):
            if any([chord[index-1]==':', chord[index-1]=='/', chord[index-1]=='(', chord[index-1]==')']):
              break
                       
            # copy character into degrees
            degrees[tempindex-1] = chord[index-1]
            index = index + 1
            tempindex = tempindex + 1
                   
          if (index > ilength):
            print 'Error in parsechord: \nDegree list brackets not closed in chord "' + chord 
            success = 0; 
                   
        else:
          print 'Error in parsechord: Incorrect character sequence in chord "' + chord 
          success = 0
          index = ilength + 1
               
      elif chord[index-1] == ')':
               
        # if we find a closing bracket then we should either be at
        # the end of the symbol or there should be a slash to follow
        if all([colon, openbracket, not slash, not closebracket]): 
          closebracket = True
          index = index + 1
                
        else:
          print 'Error in parsechord: \nIncorrect character sequence in chord "' + chord 
          success = 0
          index = ilength + 1
               
        # check to see that the brackets contained something
        if len(degrees) > 0:
          print 'Error in parsechord: Brackets contain no degrees in chord "' + chord 
          success = 0
          index = ilength + 1
                              
      elif chord[index-1] == '/':
    
        # forward slash should be followed by a degree string
              
        slash = True

        # move on to next character to process the expected bass degree
        index = index +1;
                                     
        # check that we haven't overun the end of the symbol string
        if (index > ilength):
          print 'Error in parsechord: \nNo bass degree "/" at end of chord "' + chord
          success = 0
                   
        # check that if we have had an open bracket that it also
        # had a closed bracket
        if openbracket != closebracket:
          print 'Error in parsechord: Found "/" before closing bracket in chord "' + chord
          success = 0
          index = ilength + 1
    
        # check that the previous character was not a ':'
        if (chord[index-3] == ':'):
          print 'Error in parsechord: \nFound "/" directly after ":" in chord "' + chord
          success = 0
          index = ilength + 1
                   
        while index <= ilength:
                       
          # if we find a switch character after a slash then
          # the symbol is incorrect
          if any([chord[index-1]==':', chord[index-1]=='/', chord[index-1]=='(', chord[index-1]==')']):
            print 'Error in parsechord: \nIncorrect character sequence in chord "' + chord
            success = 0
            index = ilength +1
                       
          else:
                           
            # copy remaining characters into bass
            bass = bass + chord[index-1]
            index = index + 1
            tempindex = tempindex + 1                      

      else:    
        print 'Error in parsechord: Unrecognised chord "' + chord
        success = 0
        index = ilength + 1 

  return  rootnote, shorthand, degrees, bass, success, errormessage            
  
def parsenote(note, verbose=0):

  errormessage = ''

  ilength = len(note)

  accidentals = 0;
  natural = [];
  success = True

  index = 1

  if len(note) == 0:  # if we have been passed an empty string
    success = False

  else:  # parse the note string
  
    if note[index-1] in ['A','B','C','D','E','F','G']:

      # first character should be a natural
      natural = note[index-1];
      index = index + 1

      # remaining characters should be sharps or flats
      while index <= ilength:

        if note[index-1] == 'b':
          accidentals = accidentals - 1 # decrement accidental count
          index = index + 1

        elif note[index-1] == '#':
          accidentals = accidentals + 1 # increment accidental count
          index = index + 1

        else:
          # if neither of the above then the note is incorrect
          success = False
          index = ilength + 1

    else:
      success = False        

    
  if not success: # correct note therefore return success = 1 
    # if not an integer then the note string is incorrect
    print 'Error in parsenote: Unrecognised note "' + note 

  return natural,accidentals,success, errormessage  
  
def shorthand2degrees(shorthand, verbose=0):

  degreelist = ''
  errormessage = ''

  success = 1;

  # Basic chords
  if shorthand == '':
    degreelist = ''; # empty  
  elif shorthand == 'maj':
    degreelist = '3,5' # major
  elif shorthand == 'min':
    degreelist = 'b3,5' # minor
  elif shorthand == 'dim':
    degreelist = 'b3,b5' # diminished  
  elif shorthand == 'aug':
    degreelist = '3,#5' # augmented
    
  # Seventh chords:
  elif  shorthand == 'maj7':
    degreelist = '3,5,7' # major seventh
  elif shorthand == 'min7':
    degreelist = 'b3,5,b7' # minor seventh
  elif shorthand == '7':
    degreelist = '3,5,b7' # seventh
  elif shorthand == 'dim7':
    degreelist = 'b3,b5,bb7' # diminished seventh
  elif shorthand == 'hdim7':
    degreelist = 'b3,b5,b7' # half diminished seventh
  elif shorthand == 'minmaj7':
    degreelist = 'b3,5,7' # minor (major seventh)
        
  # Sixth Chords:    
  elif shorthand == 'maj6':
    degreelist = '3,5,6' # major sixth
  elif shorthand == 'min6':
    degreelist = 'b3,5,6' # minor sixth
        
  # Ninth chords:
  elif shorthand == 'maj9':
    degreelist = '3,5,7,9' # major ninth
  elif shorthand == 'min9':
    degreelist = 'b3,5,b7,b9' # minor ninth
  elif shorthand == '9':
    degreelist = '3,5,b7,9' # ninth
    
  # Suspended:
  elif shorthand == 'sus4':
    degreelist = '4,5' # suspended fourth
  elif shorthand == 'sus2':     
    degreelist = '2,5' # suspended 2nd
    
  # Elevenths:
  elif shorthand == '11':
    degreelist = '3,5,b7,9,11' # dominant 11   
  elif shorthand == 'min11':
    degreelist = 'b3,5,b7,9,11' # minor 11
  elif shorthand == 'maj11':
    degreelist = '3,5,7,9,11' # major 11
        
  # Thirteenths:
  elif shorthand == '13':
    degreelist = '3,5,b7,9,11,13' # dominant 13 
  elif shorthand == 'min13':
    degreelist = 'b3,5,b7,9,11,13' # minor 13
  elif shorthand == 'maj13':      
    degreelist = '3,5,7,9,11,13' # major 13        
        
  else:      
    print 'Error in shorthand2degrees: Unrecognised shorthand string "' + shorthand
    success = False
        
  return degreelist, success, errormessage

def parsedegreelist(degreelist, verbose=0):

  errormessage = ''
  ilength = len(degreelist)
  index = 1
  tempindex = 1
  tempstring = ''
  success = True
  parindex = 1
  pardegrees = []

  while index <= ilength:
    
    while (degreelist[index] is not ','): 
        
      tempstring[tempindex] = degreelist[index]
      tempindex = tempindex + 1
      index = index + 1
        
      if (index > ilength):
        break
        
      if (degreelist[index] == ',') and (index == ilength):
        success = False
        print 'Error in parsedegreelist: degree list finishes with a comma "' + degreelist
    
    [pardegrees(parindex,1),pardegrees(parindex,2),pardegrees(parindex,3),ok, error] = parsedegree(tempstring);
        
    if ok:
      tempstring = ''
      tempindex = 1
      parindex = parindex + 1;
      index = index + 1;
    else:
      print 'Error in parsedegreelist: incorrect degree in list "' + degreelist
      success = False
      index = ilength + 1  

  return pardegrees, success, errormessage            

def parsedegree(degree, verbose=0):

  ilength = len(degree)

  errormessage = ''
  accidentals = 0
  interval = 0
  success = True
  present = True

  index = 1;
 
  # if the input string is not empty   
  if len(degree) == 0:

    # check for omit degree '*'
    if degree[index] == '*': 
      present = 0
      index = index + 1

    tempstring = ''
    tempindex = 1
    # parse the degree string
    while index <= ilength:

      if degree[index] == 'b': # FLAT
        accidentals = accidentals - 1 #decrement accidental count
        index = index + 1

      elif degree[index] == '#': # SHARP
        accidentals = accidentals + 1 #increment accidental count
        index = index + 1
             
      elif degree[index] in ['1','2','3','4','5','6','7','8','9']:
        # if neither of the above then remaining string should be
        # an integer interval value
        tempstring[tempindex] = degree[index]
                
        tempindex = tempindex + 1
        index = index + 1
                
      else:
        # unrecognised symbol
        success = False
        index = ilength + 1

  else:
    success = False

  if success:

    # convert the interval string to an integer
    [interval, success] = str(tempstring)

    # check it worked and that the interval is valid
    if len(interval) == 0 or (interval <= 0): 
        success = 0;            

  if not success: #ß correct degree therefore return success = 1 
    # if not an integer then the degree string is incorrect
    print 'Error in parsedegree: Unrecognised degree "' + degree
    interval = 0

  return interval,accidentals,present,success, errormessage
      