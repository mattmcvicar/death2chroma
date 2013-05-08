def reduce_chords(chords,alphabet):
  # Reduce a list of chords to a simpler alphabet

  reduced_chords = chords
  if alphabet == 'minmaj':
    reduced_chords = reduce_to_minmaj(chords)
  if alphabet == 'triads':
    reduced_chords = reduce_to_triads(chords)
  if alphabet == 'quads':
    reduced_chords = reduce_to_quads(chords)

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
  
def reduce_to_triads(chords):
    
  reduced_chords = []

  for chord in (chords):
    
    quality, success = chord2quality(chord)
  
    if quality == 0:
      c_type='maj'
    elif quality == 1: 
      c_type='min'    
    elif quality == 2:
      c_type='dim'    
    elif quality == 3:
      c_type = 'aug'
    elif quality == 4:  
      c_type = 'sus'
    else:   
      # unknown
      print 'Error in reduce_to_triads: Unknown chord quality' 
      
    # get rootnote and append type
    [rootnote, shorthand,degreelist,bassdegree, success] = getchordinfo(chord)
    
    reduced_chords.append(rootnote + ':' + c_type)

  return reduced_chords  
  
def reduce_to_quads(chords):
    
  reduced_chords = []  
  for chord in chords:  
      
    chordnotes, bassnote, success = chord2notes(chord)
    
    # If it's NC, move on.
    if chord == 'N':
        reduced_chords.append('N')
        continue
    
    
    # look at the root and go one or two back to look for 7ths
    [rootnote, shorthand,degreelist,bassdegree, success] = getchordinfo(chord)
    
    minor7th = degree2note('b7',rootnote)
    major7th = degree2note('7',rootnote)
    
    min7flag = False
    maj7flag = False
    # search for them in chordnotes
    if minor7th[0] in chordnotes:
        min7flag = True
    
    if major7th[0] in chordnotes:
        maj7flag = True
    
    # Now get the rest of the chord
    quality,success = chord2quality(chord)
    
    if quality == 0:
      c_type='maj'
    if quality == 1:
      c_type='min'    
    if quality == 2:
      c_type='dim'   
    if quality == 3:
      c_type='aug'
    if quality == 4:
      c_type='sus'         
             
    # Get the degreelist for the triad
    degs, success = shorthand2degrees(c_type)
    triad_type = rootnote + ':(' + degs
    
    # see if min7 or maj7 have been flagged, else close bracket
    if min7flag:
        reduced_chords.append(triad_type + ',b7)')
    elif maj7flag:
        reduced_chords.append(triad_type + ',7)')
    else:
        reduced_chords.append(triad_type + ')')
        
  return reduced_chords
    
# Low level functions for extracting notes etc
def chord2quality(chordsymbol):

  quality = 0

  [rootnote,shorthand,degreelist,bassdegree,success] = getchordinfo(chordsymbol)

  if success:  
    if len(shorthand) > 0:  
        quality, success = short2quality(shorthand)   
    else:
        quality, success = degrees2quality(degreelist)
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
  degrees = ''
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
  # parse the chord symbol

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
            degrees = degrees + chord[index-1]
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
    print 'Error in parsenote: Unrecognised note "' + note + '"'

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
  elif shorthand == 'sus':     
    degreelist = '2,4' # suspended 2nd and 4th
    
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
    
    interval,accidentals,present,ok = parsedegree(tempstring)
    pardegrees.append([interval,accidentals,present])
        
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
    if len(interval) == 0 or (int(interval) <= 0): 
        success = False          

  if not success: # correct degree therefore return success = 1 
    # if not an integer then the degree string is incorrect
    print 'Error in parsedegree: Unrecognised degree "' + degree
    interval = False

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

def chord2notes(chordsymbol):
  
  mainlist = ''
  success = True
  chordnotes = []
  bassnote = ''

  # parse the chordsymbol
  [rootnote,shorthand,degreelist,bass, success] = getchordinfo(chordsymbol)

  # if 'no chord' then return N
  if success:
      
    if rootnote == 'N':
        chordnotes = []
    else:
        # combine shorthand and degreelist and obtain note names for each
        # degree
        if success:
            mainlist, success = addshort2list(shorthand, degreelist)

            if success:
                # convert list of degrees to list of notes 
                chordnotes,success = degrees2notes(mainlist,rootnote)

        # Now find the bass note
        if success:
            if len(bass) > 0:
              bassnote,success = degree2note(bass, rootnote)
              if success:
                ilength = len(chordnotes)
                index = 1
                while index <= ilength:
                        
                  # add an extra loop to check the notes of different
                  # size
                  for note in chordnotes:
                    if bassnote == note:
                      index = ilength + 1
                      contains = True
                      break
                    else:
                      contains = False
                      index = index + 1 
                    
                if not contains:
                  #chordnotes = [bassnote, chordnotes]
                  chordnotes.insert(0,bassnote)
            else:
                bassnote = rootnote
  return chordnotes, bassnote, success                
    
def addshort2list(shorthand, degreelist):
  
  # initialise variables
  fulldegreelist = ''
  shortintervals = []
  degreeintervals = []
  success = True
  
  # convert shorthand string to list of degrees
  shortdegrees, ok  = shorthand2degrees(shorthand)
  
  # BAH. Need to initialise a list 
  addlist = []
  n_intervals = 24
  n_max_sharps = 2
  for i in range(n_intervals + n_max_sharps):
    addlist.append([0,0])    

  if ok:
    # if that works convert the shorthand degrees and degreelist to
    # intervals and accidentals
    
    # add the implied 1 interval to the shorthand list
    if len(shortdegrees) == 0:
      rootinterval = '1' # no comma if shorthand is empty
    else:
      rootinterval = '1,' # comma shows that there are further degrees after this one

    shortintervals, ok = parsedegreelist(rootinterval + shortdegrees) 
    
    if ok:

      # this part of the algorithm finds all the present intervals and sorts
      # them into ascending numeric order
  
      # First put the shorthand intervals into the full list
      
      ilength = len(shortintervals)
   
      for index in range(ilength):
          # convert interval to a semitone value  
          semitone,success = interval2semitone(shortintervals[index][0],shortintervals[index][1]);
          
          # Using the semitone value as an index will find redundant
          # degrees and also sort them in number order. Multiplying the interval by its
          # presence value leaves either the interval value or 0. Any element 
          # in the array with interval of 0 will be discarded later
          # We add 1 to the semitone value to handle the '1'  or 'unity'
          # interval that would return a 0 semitone value
                    
          # record the interval
          if shortintervals[index][2]:
            addlist[semitone][0] = shortintervals[index][0]
          else:
            addlist[semitone][0] = 0    
         
          # record the accidentals as well
          addlist[semitone][1] = shortintervals[index][1]
    else:
        success = 0;
        
    degreeintervals, ok2 = parsedegreelist(degreelist)
    ok = ok and ok2
    
    if ok:
      # Now add the degrees from the degreelist taking redundant and 
      # ommitted degrees into account 
      
      ilength = len(degreeintervals)
   
      for index in range(ilength):
        # convert interval to a semitone value  
        semitone, ok = interval2semitone(degreeintervals[index][0],degreeintervals[index][1])
          
        if (ok and semitone >= 0):
          # record the interval 
          if degreeintervals[index][2]:
            addlist[semitone][0] = degreeintervals[index][0]
          else:
            addlist[semitone][0] = 0  
          
          # record the accidentals as well
          addlist[semitone][1] = degreeintervals[index][1]
          
      else:
        success = False 
    else:
        success = False
    
    # now create fulldegreelist
    if ok:

        ilength = len(addlist)

        for index in range(ilength):

          # if there is a non zero interval in this element, convert 
          # it to a degree symbol and add it to the output list
          if (addlist[index][0] !=0):

            degreesymbol,success = interval2degree(addlist[index][0],addlist[index][1])

            if len(fulldegreelist) == 0:
              # if this is the first degree then start list
              fulldegreelist = degreesymbol
            else:
              # add to list with a separating comma
              fulldegreelist = fulldegreelist + ',' + degreesymbol
              
    else:
        success = False
            
  else:
    success = False
    
  return fulldegreelist, success  
    
def interval2degree(interval,accidentals):

  import numpy as np
  # set verbose default to 0
  success = True
  degree = ''

  if accidentals >= 1:
    # then the interval is either a natural or has a number of sharps 
    if accidentals != 0:
      for index in range(accidentals):
        degree = degree + '#'
    
  else:
    # then the interval has a number of flats 
    abs_accidentals = np.abs(accidentals)
    for index in range(abs_accidentals):   
        degree = degree + 'b'

  if type(interval) is str:  
    degree = degree + interval
  else:
   success = False
 
  return degree,success

def degrees2notes(degreelist, root):

  ilength = len(degreelist)
  index = 1
  tempindex = 1
  tempstring = ''
  success = True
  notes = []

  while index <= ilength:
    
    while (degreelist[index-1] != ','):  
        
      tempstring = tempstring + degreelist[index-1]
      tempindex = tempindex + 1
      index = index + 1
        
      if (index > ilength):
            break
        
      if (degreelist[index-1] == ',') and (index == ilength):
        success = False
        print('Error in degrees2notes: degree list finishes with a comma "' + degreelist)
    
    newnote,ok = degree2note(tempstring,root)
  
    if ok:
      tempstring = ''
      tempindex = 1
      notes.append(newnote)
      index = index + 1;
    else:
      print('Error in degrees2notes: incorrect degree in list "' + degreelist)
      success = False
      index = ilength + 1 

  return notes, success  

def degree2note(degree, root):

  import numpy as np
  note = [];

  intervaltranslation = [5,0,2,4,-1,1,3,5]; # scale degree translations on line of fifths 

  fifthpositions = ['F','C','G','D','A','E','B']; #order of naturals on line of fifths

  success = True

  # parse the root note
  [rootnatural,rootaccs, rsuccess] = parsenote(root);

  # parse the degree
  [interval, degreeaccs, present, dsuccess] = parsedegree(degree)

  # if parsing symbols was successful
  if (rsuccess and dsuccess):

    if rootnatural == 'F':
      fifthindex = 0;        
    if rootnatural == 'C':
      fifthindex = 1;
    if rootnatural == 'G':
      fifthindex = 2;      
    if rootnatural == 'D':
      fifthindex = 3;
    if rootnatural == 'A':
      fifthindex = 4;
    if rootnatural == 'E':
      fifthindex = 5;      
    if rootnatural == 'B':
      fifthindex = 6;
      
    # locate enharmonic root on line of fifths (modulo 6 arithmetic)     
    fifthoffset = rootaccs*7
    fifthindex = fifthindex + fifthoffset;

    # calculate interval translation on line of fifths (DONT add 1 to account
    # for matlab referencing of array elements... 
    
    intervaloffset = intervaltranslation[np.mod(int(interval),7)]
    finalposition = fifthindex + intervaloffset
        
    naturalvalue = np.mod(finalposition,7);
    
    # calculate number of accidentals
    if finalposition < 0: 
      # if final position is negative then calculate number of flats
      # remembering to include the extra first flat (-1)
      accidentals = np.fix((finalposition+1)/7) + degreeaccs -1    
    else:
      # note is a natural or has a number of sharps
      accidentals = int(np.fix(finalposition/7) + degreeaccs)
    
    note = fifthpositions[naturalvalue]    
    if accidentals > 0:       
        for i in range(accidentals):
            note = note + '#'
                    
    elif accidentals <= 0:
        abs_acc = int(np.abs(accidentals))
        for i in range(abs_acc):
          note = note + 'b'
        
  else: 
    success = False
    
  return note,success    
            


  