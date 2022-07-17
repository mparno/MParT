classdef MultiIndex < handle
%DATABASE Example usage of the mexplus development kit.
%
% This class definition gives an interface to the underlying MEX functions
% built in the private directory. It is a good practice to wrap MEX functions
% with Matlab script so that the API is well documented and separated from
% its C++ implementation. Also such a wrapper is a good place to validate
% input arguments.
%
% Build
% -----
%
%    make
%
% See `make.m` for details.
%

properties (Access = private)
  id_
end

methods
  function this = MultiIndex(varargin)
    if(nargin==2)
      this.id_ = MParT_('MultiIndex_newDefault', varargin{1},varargin{2});
    else
      if length(varargin{1})==1
        this.id_ = MParT_('MultiIndex_newDefault', varargin{1},0);
      else
        this.id_ = MParT_('MultiIndex_newEigen',varargin{1});
      end
    end
  end

  function delete(this)
  %DELETE Destructor.
    MParT_('MultiIndex_delete', this.id_);
  end

  function result = String(this)
    result = MParT_('MultiIndex_String', this.id_);
  end

  
end

end
