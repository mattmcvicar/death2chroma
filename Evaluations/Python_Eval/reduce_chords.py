def reduce_chords(chords,alphabet):
  # Reduce a list of chords to a simpler alphabet

  reduced_chords = chords
  if alphabet == 'minmaj':
    reduced_chords = reduce_to_minmaj(chords)
  if alphabet == 'triads':
    pass
  if alphabet == 'quads':
    pass

  return reduced_chords
  
# Mid-level functions to reduce the chords  
def reduce_to_minmaj(chords):
    
  reduced_chords = []

  for chord in (chords):
    
    quality, success = chord2quality(chord)
  
    if quality == 0:
      # major
      c_type='maj'
    elif quality == 1: 
      # major
      c_type='maj'    
    elif quality == 2:
      # major
      c_type='maj'    
    elif quality == 3:
      # major
      c_type = 'maj'
    elif quality == 4:  
      # suspended
      c_type = 'maj'
    else:   
      # unknown
      print 'Error in reduce_to_minmaj: Unknown chord quality' 
      
    # get rootnote and append type
    [rootnote, shorthand,degreelist,bassdegree, success] = getchordinfo(chord)
    
    reduced_chords.append(rootnote + ':' + c_type)

  return reduced_chords
  
# Low level functions for extracting notes etc
def chord2quality(chordsymbol):

  quality = 0

  [rootnote,shorthand,degreelist,bassdegree,success] = getchordinfo(chordsymbol)

  if success:  
    if len(shorthand) > 0:  
        [quality, success] = short2quality(shorthand)   
    else:
        [quality, success] = degrees2quality(degreelist)
  else: 
    pass    

  return  quality, success       
   
def getchordinfo(chordsymbol):
    
  rootnote = ''
  shorthand =  ''

  success = False

  # parse the chord symbol into its constituent parts
  [rootnote,shorthand, degreelist,bassdegree, success] = parsechord(chordsymbol)

  if success:
    if rootnote is not 'N':
        
        # check validity of rootnote
        [temp, temp2, success] = parsenote(rootnote)

        # check validity of shorthand list
        if success and len(shorthand) > 0:
           [temp, success] = shorthand2degrees(shorthand)
        
        # check validity of degreelist
        if success and len(degreelist) > 0:
            [temp, success] = parsedegreelist(degreelist)
    
        # check validity of bass degree
        if success and len(bassdegree) > 0:
           [temp,temp2,temp3, success] = parsedegree(bassdegree) 
    
  return rootnote, shorthand,degreelist,bassdegree, success

def parsechord(chord):

  ilength = len(chord)

  # initialise variables
  rootnote = ''
  shorthand = ''
  degrees = []
  bass = ''

  success = True
  index = 1

  # check for 'no chord' symbol
  if chord[index-1] == 'N':
    rootnote = chord[index-1]
    index = index + 1
    # check to see there are no further characters
    if (index <= ilength):
        print 'Error in parsechord: Extra characters after "no chord" symbol' 
        success = 0
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
        shorthand = 'maj'
    
    # initialise booleans to record which switch characters we have found
    colon = False
    openbracket = False
    closebracket = False
    slash = False
    
    # parse the rest of the chord symbol
    while(index <= ilength):
      
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
            degrees.append(chord[index-1])
            index = index + 1
            tempindex = tempindex + 1
                   
          if (index > ilength):
            print 'Error in parsechord: \nDegree list brackets not closed in chord "' + chord 
            success = 0 
                   
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
        if len(degrees) == 0:
          print 'Error in parsechord: Brackets contain no degrees in chord "' + chord 
          success = 0
          index = ilength + 1
                              
      elif chord[index-1] == '/':
    
        # forward slash should be followed by a degree string
        slash = True

        # move on to next character to process the expected bass degree
        index = index +1
                                     
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

  return  rootnote, shorthand, degrees, bass, success           
  
def parsenote(note):

  ilength = len(note)

  accidentals = 0
  natural = []
  success = True

  index = 1

  if len(note) == 0:  # if we have been passed an empty string
    success = False

  else:  # parse the note string
  
    if note[index-1] in ['A','B','C','D','E','F','G']:

      # first character should be a natural
      natural = note[index-1]
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

  return natural,accidentals,success
  
def shorthand2degrees(shorthand):

  degreelist = ''

  success = True

  # Basic chords
  if shorthand == '':
    degreelist = '' # empty  
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
        
  return degreelist, success

def parsedegreelist(degreelist):

  ilength = len(degreelist)
  index = 1
  tempindex = 1
  tempstring = ''
  success = True
  parindex = 1
  pardegrees = []

  while index <= ilength:
    
    while (degreelist[index-1] is not ','): 
        
      tempstring = tempstring + degreelist[index-1]
      tempindex = tempindex + 1
      index = index + 1
        
      if (index > ilength):
        break
        
      if (degreelist[index-1] == ',') and (index == ilength):
        success = False
        print 'Error in parsedegreelist: degree list finishes with a comma "' + degreelist
    
    [temp1,temp2,temp3,ok] = parsedegree(tempstring)
        
    if ok:
      tempstring = ''
      tempindex = 1
      parindex = parindex + 1
      index = index + 1
    else:
      print 'Error in parsedegreelist: incorrect degree in list "' + str(degreelist)
      success = False
      index = ilength + 1  

  return pardegrees, success            

def parsedegree(degree):

  ilength = len(degree)

  accidentals = 0
  interval = 0
  success = True
  present = True

  index = 1
 
  # if the input string is not empty   
  if len(degree) > 0:

    # check for omit degree '*'
    if degree[index-1] == '*': 
      present = 0
      index = index + 1

    tempstring = ''
    tempindex = 1
    # parse the degree string
    while index <= ilength:

      if degree[index-1] == 'b': # FLAT
        accidentals = accidentals - 1 #decrement accidental count
        index = index + 1

      elif degree[index-1] == '#': # SHARP
        accidentals = accidentals + 1 #increment accidental count
        index = index + 1
             
      elif degree[index-1] in ['1','2','3','4','5','6','7','8','9']:
        # if neither of the above then remaining string should be
        # an integer interval value
        tempstring = tempstring + degree[index-1]
                
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
    interval = str(tempstring)

    # check it worked and that the interval is valid
    if len(interval) == 0 or (interval <= 0): 
        success = 0            

  if not success: # correct degree therefore return success = 1 
    # if not an integer then the degree string is incorrect
    print 'Error in parsedegree: Unrecognised degree "' + degree
    interval = 0

  return interval, accidentals, present, success
    
def short2quality(shorthand):

  success = True
  quality = ''

  # Triads
  if shorthand == '':  
    quality = 0
  elif shorthand == 'maj':
    quality = 0 
  elif shorthand == 'min': 
    quality = 1       
  elif shorthand == 'dim':
    quality = 2   
  elif shorthand == 'aug': 
    quality = 3
      
  # sevenths    
  elif shorthand == 'maj7': 
    quality = 0
  elif shorthand == 'min7': 
    quality = 1
  elif shorthand == '7': 
    quality = 0        
  elif shorthand == 'minmaj7': 
    quality = 1        
  elif shorthand == 'dim7': 
    quality = 2
  elif shorthand == 'hdim7':
    quality = 2
 
  # sixths
  elif shorthand == 'maj6':
    quality = 0       
  elif shorthand == 'min6': 
    quality = 1
    
  # ninths 
  elif shorthand == '9':
    quality = 0          
  elif shorthand == 'maj9':
    quality = 0    
  elif shorthand == 'min9':
    quality = 1    

  # suspended
  elif shorthand == 'sus4':
    quality = 4
  elif shorthand == 'sus2':
    quality = 4
        
  # Elevenths:
  elif shorthand == '11':
    quality = 0
  elif shorthand == 'min11':
    quality = 1        
  elif shorthand == 'maj11':
    quality = 0
        
  # Thirteenths:
  elif shorthand == '13':
    quality = 0 
  elif shorthand == 'min13':
    quality = 1
  elif shorthand == 'maj13':
    quality = 0
  else:
    success = False
    print 'Error in short2quality: unrecognised shorthand: ' + shorthand
    
  return quality,success  
  
def degrees2quality(degreelist):

  quality = ''

  # define templates for the 5 different triad chords (four quality families
  #plus suspension) - weights mean maj thirds will have more effect than min
  # 3rds which in turn have more effect than 5ths and then 2nds and 4ths

  templates = []
  templates.append([1,0,0,0,6,0,0,4,0,0,0,0]) # maj
  templates.append([1,0,0,5,0,0,0,4,0,0,0,0]) # min
  templates.append([1,0,0,5,0,0,4,0,0,0,0,0]) # dim
  templates.append([1,0,0,0,6,0,0,0,4,0,0,0]) # aug
  templates.append([1,0,2,0,0,2,0,4,0,0,0,0]) # sus      
         
  # get the semitone equivalents of the degrees in the degree list             
  semitones,success = degrees2semitones(degreelist)
  
  indexa = 1

  # initialise a binary vector showing which semitones are present 
  present = [0,0,0,0,0,0,0,0,0,0,0,0]

  while (indexa <= 3) and (indexa <= len(semitones)):
    
    # for each of the first three semitones in the list make its position a
    # one in the vector 'present' 
    
    if semitones[indexa-1] < 12:
      present[semitones[indexa-1]] = 1
    indexa = indexa + 1
    
  # multiply present by the templates matrix to give a vector of scores for
  # the possible qualities
  import numpy as np
  qvector = np.dot(templates,present)

  # find maximum value from the qualities vector
  # this function benfits from the max function's picking of the first
  # maximum value if there are several equal ones so is predisposed toward 
  # major if the quality is not obvious from the input. (e.g. C:(1) returns major)  
  index = np.argmax(qvector)

  # take 1 from index to give correct enumeration (no, python indexes from 0...)
  quality = index 

  if not success: 
    print 'Error in degrees2quality: incorrect degree in list "' + str(degreelist)

  return quality,success
  
def degrees2semitones(degreelist):

  ilength = len(degreelist)
  index = 1
  tempindex = 1
  tempstring = ''
  success = True
  parindex  = 1
  semitones = []

  while index <= ilength:
    
    while (degreelist[index-1] is not ','):  
        
        tempstring = tempstring + degreelist[index-1]
        tempindex = tempindex + 1
        index = index + 1
        
        if(index > ilength):
          break
        
        if (degreelist[index-1] == ',') and (index == ilength):
          success = False
          print 'Error in degrees2semitones: degree list finishes with a comma "' + degreelist
    
    out1,ok = degree2semitone(tempstring)
    semitones.append(out1)
    if ok:
      tempstring = ''
      tempindex = 1
      parindex = parindex + 1
      index = index + 1
    else:
      print 'Error in degrees2semitones: incorrect degree in list "' + degreelist
      success = False
      index = ilength + 1      

  return semitones, success       

def degree2semitone(degree):
    
  # ilength = len(degree)
  success = False
  semitone = 0

  # parse the degree string
  [interval, accidentals, present, ok] = parsedegree(degree)

  if ok:
    # convert interval and accidentals to equivalent number of semitones 
    semitone, ok = interval2semitone(interval,accidentals)
    success = True
  else:
    print 'Error in degree2semitone: incorrect degree "' + degree
 
  return semitone, success

def interval2semitone(interval, accidentals):

  # Need it to be int
  import numpy as np
  interval = int(interval)
  
  # semitone equivalents for intervals
  #           interval   1,2,3,4,5,6,7         
  semitonetranslation = [0,2,4,5,7,9,11]

  semitone = 0

  success = True
  
  if (interval > 0) and (interval < 50):

    # semitone value is the number of semitones equivalent to the interval
    # added to the number of accidentals (sharps are positive, flats are
    # negative) and the number of octaves above the reference note to
    # account for extensions 

    des_index = int(np.mod(interval,8) + np.fix(interval/8))-1
    semitone = int(semitonetranslation[des_index] + accidentals + 12*np.fix(interval/8))

  else:
    success = False    
    print 'Error in interval2semitone: out of range interval'

  return semitone, success





  