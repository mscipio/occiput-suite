# interfile - Interfile read and write
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 

from __future__ import absolute_import, print_function

__all__ = ['FileParser', 'load', 'listmode_to_sinogram']

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import json
import os


class ParsingError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Parsing Error: " + repr(self.value)


IGNORE = ['%', '\00']
OBLIGATORY = ['!']
LINE_END = ['\r\n', '\n']
DECLARATION = ":="
COMMENT = [';']
TITLES = ['!INTERFILE']


class LineParser():
    def __init__(self, line=None, line_index='unknown'):
        self.dict = {}
        if line is not None:
            self.parse_line(line, line_index)

    def parse(self, line, line_index):
        self.line = line
        self.line_index = line_index
        self.dict = {}
        is_title = self._is_title()
        if is_title:
            return self.dict
        is_comment = self._is_comment()
        if is_comment:
            return self.dict
        segments = line.split(DECLARATION)
        if len(segments) == 1:
            return self._no_declaration()
        elif len(segments) > 2:
            raise ParsingError("Line %s contains too many '%s'. \n %s " % (str(line_index), DECLARATION, self.line))
            # parse the left hand side of the expression
        left = segments[0]
        left = self._strip_outer_spaces(left)
        left = self._strip_ignore(left)
        is_obligatory = self._is_obligatory(left)
        left = self._strip_obligatory(left)
        left, unit_measure = self._get_unit_measure(left)
        if unit_measure is not None:
            unit_measure = self._strip_outer_spaces(unit_measure)
        left = self._strip_outer_spaces(left)
        field_name = left

        # parse the right hand side of the expression
        right = segments[1]
        right = self._strip_line_end(right)
        right = self._strip_outer_spaces(right)
        is_empty = self._is_empty(right)
        if is_empty:
            data = None
            data_type = None
        else:
            data_type = self._get_data_type(right)
            data = self._get_data(right)
            if data.lower() == 'none':
                data = None

        # convert to list
        if type(data) == str and len(data) > 0 and data[0] == "{":
            data = self._parse_list(data)

        # try to convert to int and to float
        try:
            data = int(data)
        except (TypeError, ValueError):
            try:
                data = float(data)
            except (TypeError, ValueError):
                pass

        # make dictionary
        self.dict = {'name': field_name, 'value': data, 'unit': unit_measure, 'type': data_type, 'listindex': None,
                     'obligatory': is_obligatory}
        return self.dict

    def _no_declaration(self):
        """Utility function called by the parser if the line has not declaration symbol. """
        # There is no declaration sequence. Check if it is an empty line. 
        if self._is_empty(self.line):
            self.dict = {}
            return self.dict
        # If it is not an empty line, raise parsing error  
        else:
            raise ParsingError("Line %s does not contain '%s'. \n %s " % (str(self.line_index), DECLARATION, self.line))

    def _is_empty(self, s):
        """Tells whether the string is empty after replacing the characters that should be ignored and line end characters. """
        s = self._strip_ignore(s)
        s = self._strip_ignore(s)
        s = self._strip_line_end(s)
        s = self._strip_outer_spaces(s)
        return (s == '')

    def _strip_ignore(self, s):
        """Strip characters that should be ignored (listed in the global variable IGNORE). """
        s2 = s
        for st in IGNORE:
            s2 = s2.replace(st, '')
        return s2

    def _strip_obligatory(self, s):
        """Strip characters that should be ignored (listed in the global variable IGNORE). """
        s2 = s
        for st in OBLIGATORY:
            s2 = s2.replace(st, '')
        return s2

    def _strip_line_end(self, s):
        """Strip line end characters (listed in the global variable LINE_END). """
        s2 = s
        for st in LINE_END:
            s2 = s2.replace(st, '')
        return s2

    def _strip_outer_spaces(self, s):
        """Strip any white spaces to the left and to the right of the string. """
        s2 = s
        while s2.startswith(' '):
            s2 = s2[1:]
        while s2.endswith(' '):
            s2 = s2[:-1]
        return s2

    def _get_unit_measure(self, s):
        if not s.endswith(')'):
            return (s, None)
        else:
            i = s.find('(')
            if i == -1:
                raise ParsingError(
                    "The parenthesis in line %s was not opened. \n %s " % (str(self.line_index), self.line))
            unit_measure = s[i + 1:-1]
            stripped_string = s[0:i]
        return (stripped_string, unit_measure)

    def _get_data_type(self, s):
        return None

    def _get_data(self, s):
        return s

    def _is_comment(self):
        """Returns True if the line is a comment (comment sequences are listed in the global variables COMMENT)."""
        l = self._strip_outer_spaces(self.line)
        for st in COMMENT:
            if l.startswith(st):
                return True
        return False

    def _is_title(self):
        """Returns True if the line is the title of the interfile."""
        l = self._strip_outer_spaces(self.line)
        for t in TITLES:
            if t in l:
                return True
        return False

    def _is_obligatory(self, s):
        l = self._strip_outer_spaces(s)
        for st in OBLIGATORY:
            if l.startswith(st):
                return True
        return False

    def _parse_list(self, s):
        # delete bracers
        try:
            s = s.translate(None, "{},")
        except:
            s = s.translate(str.maketrans('', '', "{},"))
        list = []
        for x in s.split():
            try:
                list.append(int(x))
            except (TypeError, ValueError):
                try:
                    list.append(float(x))
                except (TypeError, ValueError):
                    list.append(x)
        return list


class FileParser():
    def __init__(self, header=None):
        self.dict = {}
        if header is not None:
            if os.path.exists(header):
                self.parse_file(header)
            else:
                try:
                    self.parse_string(header)
                except ParsingError:
                    raise ParsingError(
                        "The given string does not appear to be a valid file nor a valid header content.")

    def parse_string(self, header_string):
        self.dict = {}
        line_parser = LineParser()

        fid = StringIO()
        fid.write(header_string)
        fid.seek(0)

        # parse each line 
        line_index = 0
        for line in fid:
            line_index += 1
            line_dict = line_parser.parse(line, line_index)
            if line_dict:
                name = line_dict['name']
                line_dict.pop('name')
                self.dict[name] = line_dict
        fid.close()
        return self.dict

    def parse_file(self, header_filename):
        with open(header_filename, 'r') as fid:
            file_content = fid.read()
        return self.parse_string(file_content)

    def to_dict(self):
        return self.dict

    def to_json(self):
        return json.dumps(self.dict)

    def to_obj(self):
        class Interfile():
            pass

        I = Interfile()
        for name in self.dict.keys():
            setattr(I, name.replace(' ', '_'), self.dict.get(name))
        return I


def load(filename):
    parser = FileParser()
    return parser.parse_file(filename)


def listmode_to_sinogram(filename):
    parser = FileParser()
    dic = parser.parse_file(filename)

    n_events = int(dic['total listmode word counts']['value'])
    datafile = dic['name of data file']['value']
    datafile = datafile.replace('\\', '//')
    datafile = os.path.dirname(filename) + '//' + os.path.basename(datafile).lower()
    with open(datafile, 'r') as data_fid:
        for i in range(n_events):
            event = data_fid.read(4)
            print("event: ", event)
