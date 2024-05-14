classdef ATMOptions < TrainOptions & MapOptions
    properties (Access = public)
        maxPatience = 10;
        maxSize = 10;
        maxDegrees = MultiIndex(0);
    end
    methods
        function obj = set.maxPatience(obj,value)
            obj.maxPatience = value;
        end
        function obj = set.maxSize(obj,value)
            obj.maxSize = value;
        end
        function obj = set.maxDegrees(obj,value)
            obj.maxDegrees = value;
        end
        function optionsArray = getMexOptions(obj)
            optionsArray = getMexOptions@MapOptions(obj);
            num_MapOptions = length(optionsArray);
            optionsArray{num_MapOptions+1} = char(obj.opt_alg);
            optionsArray{num_MapOptions+2} = obj.opt_stopval;
            optionsArray{num_MapOptions+3} = obj.opt_ftol_rel;
            optionsArray{num_MapOptions+4} = obj.opt_ftol_abs;
            optionsArray{num_MapOptions+5} = obj.opt_xtol_rel;
            optionsArray{num_MapOptions+6} = obj.opt_xtol_abs;
            optionsArray{num_MapOptions+7} = obj.opt_maxeval;
            optionsArray{num_MapOptions+8} = obj.opt_maxtime;
            optionsArray{num_MapOptions+9} = obj.verbose;
            optionsArray{num_MapOptions+9+1} = obj.maxPatience;
            optionsArray{num_MapOptions+9+2} = obj.maxSize;
            optionsArray{num_MapOptions+9+3} = obj.maxDegrees.get_id();
        end
    end
end