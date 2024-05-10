function map = CreateSigmoidComponent(varargin)
    if nargin == 3
        mset = varargin{1};
        centers = varargin{2};
        options = varargin{3};
        map = ConditionalMap(mset, centers, options);
    elseif nargin == 4
        inputDim = varargin{1};
        totalOrder = varargin{2};
        centers = varargin{3};
        options = varargin{4};
        map = ConditionalMap(inputDim, totalOrder, centers, options);
    end
end